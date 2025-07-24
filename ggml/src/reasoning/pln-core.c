#include "pln-core.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

// Get high-resolution timestamp
uint64_t pln_get_timestamp_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

// Create PLN Engine
struct pln_engine* pln_engine_create(struct ggml_context* ctx) {
    struct pln_engine* engine = malloc(sizeof(struct pln_engine));
    if (!engine) return NULL;
    
    engine->ctx = ctx;
    engine->capacity = 1000;
    engine->nodes = malloc(sizeof(struct pln_node*) * engine->capacity);
    engine->node_count = 0;
    engine->next_node_id = 1;
    engine->inferences_made = 0;
    engine->total_inference_time_ns = 0;
    engine->average_confidence = 0.0f;
    
    return engine;
}

// Destroy PLN Engine
void pln_engine_destroy(struct pln_engine* engine) {
    if (!engine) return;
    
    for (size_t i = 0; i < engine->node_count; i++) {
        pln_node_destroy(engine->nodes[i]);
    }
    
    free(engine->nodes);
    free(engine);
}

// Truth Value Operations
struct pln_truth_value pln_truth_value_create(float strength, float confidence, float count) {
    struct pln_truth_value tv;
    tv.strength = fmaxf(0.0f, fminf(1.0f, strength));
    tv.confidence = fmaxf(0.0f, fminf(1.0f, confidence));
    tv.count = fmaxf(0.0f, count);
    return tv;
}

// PLN Deduction: P(A->B) ∧ P(B->C) ⇒ P(A->C)
struct pln_truth_value pln_truth_value_deduction(
    const struct pln_truth_value* tv_ab, 
    const struct pln_truth_value* tv_bc
) {
    if (!tv_ab || !tv_bc) {
        return pln_truth_value_create(0.0f, 0.0f, 0.0f);
    }
    
    // Strength: s_ab * s_bc
    float strength = tv_ab->strength * tv_bc->strength;
    
    // Confidence: combine confidences with geometric mean
    float confidence = sqrtf(tv_ab->confidence * tv_bc->confidence);
    
    // Count: minimum of the two counts
    float count = fminf(tv_ab->count, tv_bc->count);
    
    return pln_truth_value_create(strength, confidence, count);
}

// PLN Induction: Evidence aggregation for generalization
struct pln_truth_value pln_truth_value_induction(
    const struct pln_truth_value* tv_ab, 
    const struct pln_truth_value* tv_a
) {
    if (!tv_ab || !tv_a) {
        return pln_truth_value_create(0.0f, 0.0f, 0.0f);
    }
    
    // Inductive strength based on conditional probability
    float strength = tv_ab->strength;
    
    // Confidence adjusted by evidence strength
    float confidence = tv_ab->confidence * tv_a->confidence;
    
    // Combined evidence count
    float count = tv_ab->count + tv_a->count;
    
    return pln_truth_value_create(strength, confidence, count);
}

// PLN Revision: Combine evidence from multiple sources
struct pln_truth_value pln_truth_value_revision(
    const struct pln_truth_value* tv1, 
    const struct pln_truth_value* tv2
) {
    if (!tv1 || !tv2) {
        return tv1 ? *tv1 : (tv2 ? *tv2 : pln_truth_value_create(0.0f, 0.0f, 0.0f));
    }
    
    // Weighted combination based on confidence
    float w1 = tv1->confidence;
    float w2 = tv2->confidence;
    float total_weight = w1 + w2;
    
    if (total_weight < 1e-6f) {
        return pln_truth_value_create(0.5f, 0.0f, 0.0f);
    }
    
    // Weighted average of strengths
    float strength = (tv1->strength * w1 + tv2->strength * w2) / total_weight;
    
    // Combined confidence (not simply additive to avoid overconfidence)
    float confidence = fminf(1.0f, w1 + w2 - w1 * w2);
    
    // Combined count
    float count = tv1->count + tv2->count;
    
    return pln_truth_value_create(strength, confidence, count);
}

// Truth value expectation (strength * confidence)
float pln_truth_value_expectation(const struct pln_truth_value* tv) {
    if (!tv) return 0.0f;
    return tv->strength * tv->confidence;
}

// Validate truth value
bool pln_truth_value_is_valid(const struct pln_truth_value* tv) {
    if (!tv) return false;
    return tv->strength >= 0.0f && tv->strength <= 1.0f &&
           tv->confidence >= 0.0f && tv->confidence <= 1.0f &&
           tv->count >= 0.0f;
}

// Create PLN Node
struct pln_node* pln_node_create(
    struct pln_engine* engine,
    enum pln_node_type type,
    struct ggml_tensor* content,
    const char* name
) {
    if (!engine) return NULL;
    
    struct pln_node* node = malloc(sizeof(struct pln_node));
    if (!node) return NULL;
    
    node->content = content;
    node->truth_value = pln_truth_value_create(0.5f, 0.5f, 1.0f);
    node->node_type = type;
    node->incoming_capacity = 10;
    node->outgoing_capacity = 10;
    node->incoming = malloc(sizeof(struct pln_node*) * node->incoming_capacity);
    node->outgoing = malloc(sizeof(struct pln_node*) * node->outgoing_capacity);
    node->incoming_count = 0;
    node->outgoing_count = 0;
    node->node_id = engine->next_node_id++;
    
    if (name) {
        node->name = malloc(strlen(name) + 1);
        strcpy(node->name, name);
    } else {
        node->name = NULL;
    }
    
    // Add to engine
    if (engine->node_count >= engine->capacity) {
        engine->capacity *= 2;
        engine->nodes = realloc(engine->nodes, sizeof(struct pln_node*) * engine->capacity);
    }
    engine->nodes[engine->node_count++] = node;
    
    return node;
}

// Destroy PLN Node
void pln_node_destroy(struct pln_node* node) {
    if (!node) return;
    
    free(node->incoming);
    free(node->outgoing);
    free(node->name);
    free(node);
}

// Add outgoing link
void pln_node_add_outgoing(struct pln_node* from, struct pln_node* to) {
    if (!from || !to) return;
    
    if (from->outgoing_count >= from->outgoing_capacity) {
        from->outgoing_capacity *= 2;
        from->outgoing = realloc(from->outgoing, sizeof(struct pln_node*) * from->outgoing_capacity);
    }
    
    from->outgoing[from->outgoing_count++] = to;
}

// Add incoming link
void pln_node_add_incoming(struct pln_node* to, struct pln_node* from) {
    if (!to || !from) return;
    
    if (to->incoming_count >= to->incoming_capacity) {
        to->incoming_capacity *= 2;
        to->incoming = realloc(to->incoming, sizeof(struct pln_node*) * to->incoming_capacity);
    }
    
    to->incoming[to->incoming_count++] = from;
}

// Find node by name
struct pln_node* pln_node_find_by_name(struct pln_engine* engine, const char* name) {
    if (!engine || !name) return NULL;
    
    for (size_t i = 0; i < engine->node_count; i++) {
        if (engine->nodes[i]->name && strcmp(engine->nodes[i]->name, name) == 0) {
            return engine->nodes[i];
        }
    }
    return NULL;
}

// Deduction Rule Implementation: P(A->B) ∧ P(B->C) ⇒ P(A->C)
struct pln_inference_result* pln_deduction(
    struct pln_engine* engine,
    struct pln_node* premise_ab,
    struct pln_node* premise_bc
) {
    if (!engine || !premise_ab || !premise_bc) return NULL;
    
    uint64_t start_time = pln_get_timestamp_ns();
    
    struct pln_inference_result* result = malloc(sizeof(struct pln_inference_result));
    if (!result) return NULL;
    
    // Create conclusion A->C
    result->conclusion = pln_node_create(engine, PLN_NODE_IMPLICATION_LINK, NULL, "deduction_result");
    
    // Calculate truth value using deduction formula
    result->truth_value = pln_truth_value_deduction(&premise_ab->truth_value, &premise_bc->truth_value);
    result->conclusion->truth_value = result->truth_value;
    
    result->rule_used = PLN_RULE_DEDUCTION;
    result->confidence_level = result->truth_value.confidence;
    
    // Store premises
    result->premise_count = 2;
    result->premises = malloc(sizeof(struct pln_node*) * 2);
    result->premises[0] = premise_ab;
    result->premises[1] = premise_bc;
    
    // Update engine statistics
    engine->inferences_made++;
    engine->total_inference_time_ns += pln_get_timestamp_ns() - start_time;
    engine->average_confidence = (engine->average_confidence * (engine->inferences_made - 1) + 
                                  result->confidence_level) / engine->inferences_made;
    
    return result;
}

// Induction Rule Implementation
struct pln_inference_result* pln_induction(
    struct pln_engine* engine,
    struct pln_node* evidence_ab,
    struct pln_node* evidence_a
) {
    if (!engine || !evidence_ab || !evidence_a) return NULL;
    
    uint64_t start_time = pln_get_timestamp_ns();
    
    struct pln_inference_result* result = malloc(sizeof(struct pln_inference_result));
    if (!result) return NULL;
    
    // Create inductive conclusion
    result->conclusion = pln_node_create(engine, PLN_NODE_INHERITANCE_LINK, NULL, "induction_result");
    
    // Calculate truth value using induction formula
    result->truth_value = pln_truth_value_induction(&evidence_ab->truth_value, &evidence_a->truth_value);
    result->conclusion->truth_value = result->truth_value;
    
    result->rule_used = PLN_RULE_INDUCTION;
    result->confidence_level = result->truth_value.confidence;
    
    // Store premises
    result->premise_count = 2;
    result->premises = malloc(sizeof(struct pln_node*) * 2);
    result->premises[0] = evidence_ab;
    result->premises[1] = evidence_a;
    
    // Update engine statistics
    engine->inferences_made++;
    engine->total_inference_time_ns += pln_get_timestamp_ns() - start_time;
    engine->average_confidence = (engine->average_confidence * (engine->inferences_made - 1) + 
                                  result->confidence_level) / engine->inferences_made;
    
    return result;
}

// Abduction Rule Implementation
struct pln_inference_result* pln_abduction(
    struct pln_engine* engine,
    struct pln_node* rule_ab,
    struct pln_node* evidence_b
) {
    if (!engine || !rule_ab || !evidence_b) return NULL;
    
    uint64_t start_time = pln_get_timestamp_ns();
    
    struct pln_inference_result* result = malloc(sizeof(struct pln_inference_result));
    if (!result) return NULL;
    
    // Create abductive hypothesis A
    result->conclusion = pln_node_create(engine, PLN_NODE_CONCEPT, NULL, "abduction_hypothesis");
    
    // Abductive reasoning: if A->B and B, then possibly A
    // Strength is reduced due to logical fallacy (affirming consequent)
    float strength = rule_ab->truth_value.strength * evidence_b->truth_value.strength * 0.7f; // Penalty for uncertainty
    float confidence = fminf(rule_ab->truth_value.confidence, evidence_b->truth_value.confidence) * 0.8f;
    float count = fminf(rule_ab->truth_value.count, evidence_b->truth_value.count);
    
    result->truth_value = pln_truth_value_create(strength, confidence, count);
    result->conclusion->truth_value = result->truth_value;
    
    result->rule_used = PLN_RULE_ABDUCTION;
    result->confidence_level = result->truth_value.confidence;
    
    // Store premises
    result->premise_count = 2;
    result->premises = malloc(sizeof(struct pln_node*) * 2);
    result->premises[0] = rule_ab;
    result->premises[1] = evidence_b;
    
    // Update engine statistics
    engine->inferences_made++;
    engine->total_inference_time_ns += pln_get_timestamp_ns() - start_time;
    engine->average_confidence = (engine->average_confidence * (engine->inferences_made - 1) + 
                                  result->confidence_level) / engine->inferences_made;
    
    return result;
}

// Revision Rule Implementation
struct pln_inference_result* pln_revision(
    struct pln_engine* engine,
    struct pln_node* belief1,
    struct pln_node* belief2
) {
    if (!engine || !belief1 || !belief2) return NULL;
    
    uint64_t start_time = pln_get_timestamp_ns();
    
    struct pln_inference_result* result = malloc(sizeof(struct pln_inference_result));
    if (!result) return NULL;
    
    // Create revised belief
    result->conclusion = pln_node_create(engine, belief1->node_type, NULL, "revised_belief");
    
    // Calculate truth value using revision formula
    result->truth_value = pln_truth_value_revision(&belief1->truth_value, &belief2->truth_value);
    result->conclusion->truth_value = result->truth_value;
    
    result->rule_used = PLN_RULE_REVISION;
    result->confidence_level = result->truth_value.confidence;
    
    // Store premises
    result->premise_count = 2;
    result->premises = malloc(sizeof(struct pln_node*) * 2);
    result->premises[0] = belief1;
    result->premises[1] = belief2;
    
    // Update engine statistics
    engine->inferences_made++;
    engine->total_inference_time_ns += pln_get_timestamp_ns() - start_time;
    engine->average_confidence = (engine->average_confidence * (engine->inferences_made - 1) + 
                                  result->confidence_level) / engine->inferences_made;
    
    return result;
}

// Destroy inference result
void pln_inference_result_destroy(struct pln_inference_result* result) {
    if (!result) return;
    free(result->premises);
    free(result);
}

// Get inference rate (inferences per second)
float pln_engine_get_inference_rate(struct pln_engine* engine) {
    if (!engine || engine->total_inference_time_ns == 0) return 0.0f;
    
    double seconds = (double)engine->total_inference_time_ns / 1e9;
    return (float)engine->inferences_made / seconds;
}

// Reset statistics
void pln_engine_reset_stats(struct pln_engine* engine) {
    if (!engine) return;
    engine->inferences_made = 0;
    engine->total_inference_time_ns = 0;
    engine->average_confidence = 0.0f;
}

// Integration with existing hypergraph memory
struct pln_node* pln_from_hypergraph_node(
    struct pln_engine* engine,
    void* hypergraph_node_ptr
) {
    if (!engine || !hypergraph_node_ptr) return NULL;
    
    // This is a placeholder for integration with the existing hypergraph_node structure
    // In the real implementation, this would convert the hypergraph_node to a pln_node
    // For now, we create a basic PLN node
    
    struct pln_node* pln_node = pln_node_create(engine, PLN_NODE_CONCEPT, NULL, "from_hypergraph");
    pln_node->truth_value = pln_truth_value_create(0.8f, 0.7f, 1.0f); // Default values
    
    return pln_node;
}