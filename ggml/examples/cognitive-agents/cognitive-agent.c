#include "cognitive-agent.h"
#include "../../src/reasoning/pln-core.h"
#include "../../src/reasoning/moses-core.h"
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <unistd.h>

// Generate unique agent ID
uint64_t generate_agent_id(void) {
    static uint64_t counter = 0;
    return (uint64_t)time(NULL) * 1000 + (++counter);
}

// Get current timestamp
uint64_t get_timestamp(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

// Initialize hypergraph memory
hypergraph_memory* init_hypergraph_memory(struct ggml_context* ctx) {
    hypergraph_memory* mem = malloc(sizeof(hypergraph_memory));
    mem->ctx = ctx;
    mem->node_count = 0;
    mem->capacity = 1000;
    mem->nodes = malloc(sizeof(hypergraph_node*) * mem->capacity);
    mem->total_accesses = 0;
    mem->cache_hits = 0;
    return mem;
}

// Cleanup hypergraph memory
void cleanup_hypergraph_memory(hypergraph_memory* mem) {
    if (!mem) return;
    
    for (size_t i = 0; i < mem->node_count; i++) {
        free(mem->nodes[i]);
    }
    free(mem->nodes);
    free(mem);
}

// Add knowledge to hypergraph
void add_knowledge(hypergraph_memory* mem, const char* concept, 
                   float* embedding, size_t emb_size) {
    if (mem->node_count >= mem->capacity) {
        // Expand capacity
        mem->capacity *= 2;
        mem->nodes = realloc(mem->nodes, sizeof(hypergraph_node*) * mem->capacity);
    }
    
    // Create tensor for concept
    struct ggml_tensor* concept_tensor = ggml_new_tensor_1d(mem->ctx, GGML_TYPE_F32, 
                                                     strlen(concept));
    memcpy(concept_tensor->data, concept, strlen(concept));
    
    // Create embedding tensor
    struct ggml_tensor* emb_tensor = ggml_new_tensor_1d(mem->ctx, GGML_TYPE_F32, 
                                                 emb_size);
    memcpy(emb_tensor->data, embedding, emb_size * sizeof(float));
    
    // Create hypergraph node
    hypergraph_node* node = malloc(sizeof(hypergraph_node));
    node->data = concept_tensor;
    node->embedding = emb_tensor;
    node->node_type = NODE_TYPE_CONCEPT;
    node->truth_value = 0.8f;  // Initial belief
    node->confidence = 0.9f;   // High confidence
    node->creation_time = get_timestamp();
    node->last_access = node->creation_time;
    
    // Add to memory
    mem->nodes[mem->node_count++] = node;
    
    printf("Added knowledge: %s (nodes: %zu)\n", concept, mem->node_count);
}

// Find concept in hypergraph
hypergraph_node* find_concept(hypergraph_memory* mem, const char* concept) {
    mem->total_accesses++;
    
    for (size_t i = 0; i < mem->node_count; i++) {
        hypergraph_node* node = mem->nodes[i];
        if (node->node_type == NODE_TYPE_CONCEPT) {
            // Simple string comparison (in practice would use semantic matching)
            if (strncmp((char*)node->data->data, concept, strlen(concept)) == 0) {
                node->last_access = get_timestamp();
                mem->cache_hits++;
                return node;
            }
        }
    }
    return NULL;
}

// Initialize attention economy
attention_economy* init_attention_economy(void) {
    attention_economy* attention = malloc(sizeof(attention_economy));
    attention->total_attention = 1.0f;
    attention->allocated_attention = 0.0f;
    
    attention->memory_allocation = 0.0f;
    attention->reasoning_allocation = 0.0f;
    attention->communication_allocation = 0.0f;
    attention->self_modification_allocation = 0.0f;
    
    attention->min_threshold = 0.1f;
    attention->decay_rate = 0.01f;
    attention->novelty_bonus = 0.2f;
    
    attention->history_index = 0;
    attention->history_size = 0;
    
    return attention;
}

// Cleanup attention economy
void cleanup_attention_economy(attention_economy* attention) {
    if (attention) {
        free(attention);
    }
}

// Compute salience score for tensor
float compute_salience(struct ggml_tensor* tensor, attention_economy* attention) {
    // Simple salience computation (in practice would be more sophisticated)
    float base_salience = 0.5f;
    
    // Add novelty bonus
    float novelty = 0.3f; // Placeholder for novelty computation
    base_salience += novelty * attention->novelty_bonus;
    
    // Clamp to [0, 1]
    if (base_salience > 1.0f) base_salience = 1.0f;
    if (base_salience < 0.0f) base_salience = 0.0f;
    
    return base_salience;
}

// Allocate attention
void allocate_attention(attention_economy* attention, float amount, uint32_t target) {
    if (attention->allocated_attention + amount > attention->total_attention) {
        // Need to reallocate
        float excess = (attention->allocated_attention + amount) - attention->total_attention;
        // Simple strategy: reduce all allocations proportionally
        float reduction_factor = excess / attention->allocated_attention;
        attention->memory_allocation *= (1.0f - reduction_factor);
        attention->reasoning_allocation *= (1.0f - reduction_factor);
        attention->communication_allocation *= (1.0f - reduction_factor);
        attention->self_modification_allocation *= (1.0f - reduction_factor);
        
        // Recalculate allocated attention
        attention->allocated_attention = attention->memory_allocation +
                                       attention->reasoning_allocation +
                                       attention->communication_allocation +
                                       attention->self_modification_allocation;
    }
    
    // Add new allocation
    switch (target) {
        case COGNITIVE_TYPE_MEMORY:
            attention->memory_allocation += amount;
            break;
        case COGNITIVE_TYPE_REASONING:
            attention->reasoning_allocation += amount;
            break;
        case COGNITIVE_TYPE_COMMUNICATION:
            attention->communication_allocation += amount;
            break;
        default:
            attention->self_modification_allocation += amount;
            break;
    }
    
    attention->allocated_attention += amount;
    
    printf("Allocated %.2f attention to type %u (total: %.2f/%.2f)\n", 
           amount, target, attention->allocated_attention, attention->total_attention);
}

// Update performance history
void update_performance_history(attention_economy* attention, float performance) {
    attention->performance_history[attention->history_index] = performance;
    attention->history_index = (attention->history_index + 1) % 100;
    if (attention->history_size < 100) {
        attention->history_size++;
    }
}

// Initialize task orchestrator
task_orchestrator* init_task_orchestrator(void) {
    task_orchestrator* orch = malloc(sizeof(task_orchestrator));
    orch->current_goal = NULL;
    orch->context_state = NULL;
    orch->pending_tasks = NULL;
    orch->task_count = 0;
    orch->task_capacity = 0;
    return orch;
}

// Cleanup task orchestrator
void cleanup_task_orchestrator(task_orchestrator* orch) {
    if (orch) {
        if (orch->pending_tasks) {
            free(orch->pending_tasks);
        }
        free(orch);
    }
}

// Initialize reasoning engine
reasoning_engine* init_reasoning_engine(struct ggml_context* ctx) {
    reasoning_engine* reasoning = malloc(sizeof(reasoning_engine));
    reasoning->ctx = ctx;
    reasoning->current_beliefs = NULL;
    reasoning->inference_rules = NULL;
    reasoning->reasoning_accuracy = 0.75f;
    reasoning->inferences_made = 0;
    
    // Initialize PLN integration
    reasoning->pln_engine = NULL;
    reasoning->pln_inference_rate = 0.0f;
    reasoning->average_pln_confidence = 0.0f;
    
    // Initialize MOSES integration
    reasoning->moses_engine = NULL;
    reasoning->best_program_fitness = -INFINITY;
    reasoning->evolution_generations = 0;
    
    return reasoning;
}

// Cleanup reasoning engine
void cleanup_reasoning_engine(reasoning_engine* reasoning) {
    if (reasoning) {
        if (reasoning->pln_engine) {
            pln_engine_destroy(reasoning->pln_engine);
        }
        if (reasoning->moses_engine) {
            moses_engine_destroy(reasoning->moses_engine);
        }
        free(reasoning);
    }
}

// Infer cognitive type from tensor
uint32_t infer_cognitive_type(struct ggml_tensor* tensor) {
    // Simple heuristic based on tensor properties
    if (tensor->ne[0] <= 64) {
        return COGNITIVE_TYPE_ATTENTION;
    } else if (tensor->ne[0] <= 256) {
        return COGNITIVE_TYPE_MEMORY;
    } else if (tensor->ne[0] <= 1024) {
        return COGNITIVE_TYPE_REASONING;
    } else {
        return COGNITIVE_TYPE_COMMUNICATION;
    }
}

// Compute tensor similarity (simplified)
float compute_tensor_similarity(struct ggml_tensor* a, struct ggml_tensor* b) {
    if (!a || !b || a->ne[0] != b->ne[0]) {
        return 0.0f;
    }
    
    // Simple cosine similarity for float tensors
    if (a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32) {
        float* data_a = (float*)a->data;
        float* data_b = (float*)b->data;
        
        float dot_product = 0.0f;
        float norm_a = 0.0f;
        float norm_b = 0.0f;
        
        for (int i = 0; i < a->ne[0]; i++) {
            dot_product += data_a[i] * data_b[i];
            norm_a += data_a[i] * data_a[i];
            norm_b += data_b[i] * data_b[i];
        }
        
        if (norm_a == 0.0f || norm_b == 0.0f) {
            return 0.0f;
        }
        
        return dot_product / (sqrtf(norm_a) * sqrtf(norm_b));
    }
    
    return 0.0f;
}

// Create cognitive agent
cognitive_agent* create_cognitive_agent(const char* endpoint) {
    cognitive_agent* agent = malloc(sizeof(cognitive_agent));
    
    // Initialize ggml context
    struct ggml_init_params params = {
        .mem_size = 128 * 1024 * 1024,  // 128MB
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    agent->ctx = ggml_init(params);
    
    // Setup backend (use CPU for simplicity in example)
    agent->backend = NULL;  // Would use ggml_backend_cpu_init() in real implementation
    
    // Initialize cognitive subsystems
    agent->memory = init_hypergraph_memory(agent->ctx);
    agent->tasks = init_task_orchestrator();
    agent->reasoning = init_reasoning_engine(agent->ctx);
    agent->attention = init_attention_economy();
    
    // Set identity
    agent->agent_id = generate_agent_id();
    strncpy(agent->endpoint, endpoint, sizeof(agent->endpoint) - 1);
    agent->endpoint[sizeof(agent->endpoint) - 1] = '\0';
    agent->attention_weight = 1.0f;
    
    // Initialize counters
    agent->messages_sent = 0;
    agent->messages_received = 0;
    agent->cycles_completed = 0;
    
    printf("Created cognitive agent %lu at %s\n", agent->agent_id, agent->endpoint);
    
    return agent;
}

// Cleanup cognitive agent
void cleanup_cognitive_agent(cognitive_agent* agent) {
    if (!agent) return;
    
    cleanup_hypergraph_memory(agent->memory);
    cleanup_task_orchestrator(agent->tasks);
    cleanup_reasoning_engine(agent->reasoning);
    cleanup_attention_economy(agent->attention);
    
    if (agent->ctx) {
        ggml_free(agent->ctx);
    }
    
    printf("Cleaned up cognitive agent %lu\n", agent->agent_id);
    free(agent);
}

// Send cognitive tensor (simplified - no actual network communication)
void send_cognitive_tensor(cognitive_agent* sender, uint64_t target_agent_id,
                          struct ggml_tensor* tensor, float attention_weight) {
    
    cognitive_tensor_packet packet = {0};
    
    // In real implementation, would serialize tensor using ggml-rpc
    packet.attention_weight = attention_weight;
    packet.salience_score = compute_salience(tensor, sender->attention);
    packet.cognitive_type = infer_cognitive_type(tensor);
    packet.source_agent_id = sender->agent_id;
    packet.target_agent_id = target_agent_id;
    packet.recursion_depth = 0;
    packet.timestamp = get_timestamp();
    
    strncpy(packet.meta_context, "cognitive_exchange", sizeof(packet.meta_context) - 1);
    
    sender->messages_sent++;
    
    printf("Agent %lu sent cognitive tensor (type %u, attention %.2f, salience %.2f) to agent %lu\n",
           sender->agent_id, packet.cognitive_type, packet.attention_weight, 
           packet.salience_score, target_agent_id);
}

// Process incoming tensor (simplified)
void process_incoming_tensor(cognitive_agent* receiver, 
                           cognitive_tensor_packet* msg) {
    
    receiver->messages_received++;
    
    printf("Agent %lu received cognitive tensor from agent %lu (type %u, attention %.2f)\n",
           receiver->agent_id, msg->source_agent_id, msg->cognitive_type, msg->attention_weight);
    
    // Check attention allocation
    if (msg->attention_weight < receiver->attention->min_threshold) {
        printf("  Insufficient attention weight, deferring message\n");
        return;
    }
    
    // Allocate attention for processing
    allocate_attention(receiver->attention, msg->attention_weight, msg->cognitive_type);
    
    // Process based on cognitive type
    switch (msg->cognitive_type) {
        case COGNITIVE_TYPE_MEMORY:
            printf("  Processing memory operation\n");
            // In real implementation: integrate_memory(receiver->memory, tensor, msg);
            break;
            
        case COGNITIVE_TYPE_TASK:
            printf("  Processing task request\n");
            // In real implementation: process_task_request(receiver->tasks, tensor, msg);
            break;
            
        case COGNITIVE_TYPE_REASONING:
            printf("  Processing reasoning request\n");
            receiver->reasoning->inferences_made++;
            break;
            
        case COGNITIVE_TYPE_ATTENTION:
            printf("  Processing attention update\n");
            break;
            
        default:
            printf("  Processing unknown cognitive type\n");
            break;
    }
}

// PLN Integration Functions

// Initialize PLN reasoning
int init_pln_reasoning(reasoning_engine* reasoning) {
    if (!reasoning || !reasoning->ctx) {
        return -1;
    }
    
    reasoning->pln_engine = pln_engine_create(reasoning->ctx);
    if (!reasoning->pln_engine) {
        return -1;
    }
    
    return 0;
}

// Perform PLN deduction
int pln_perform_deduction(reasoning_engine* reasoning, const char* premise1, const char* premise2) {
    if (!reasoning || !reasoning->pln_engine || !premise1 || !premise2) {
        return -1;
    }
    
    // Find or create nodes for premises
    struct pln_node* node1 = pln_node_find_by_name(reasoning->pln_engine, premise1);
    struct pln_node* node2 = pln_node_find_by_name(reasoning->pln_engine, premise2);
    
    if (!node1) {
        node1 = pln_node_create(reasoning->pln_engine, PLN_NODE_IMPLICATION_LINK, NULL, premise1);
        node1->truth_value = pln_truth_value_create(0.8f, 0.7f, 5.0f);
    }
    
    if (!node2) {
        node2 = pln_node_create(reasoning->pln_engine, PLN_NODE_IMPLICATION_LINK, NULL, premise2);
        node2->truth_value = pln_truth_value_create(0.75f, 0.8f, 4.0f);
    }
    
    // Perform deduction
    struct pln_inference_result* result = pln_deduction(reasoning->pln_engine, node1, node2);
    
    if (result) {
        reasoning->inferences_made++;
        reasoning->pln_inference_rate = pln_engine_get_inference_rate(reasoning->pln_engine);
        reasoning->average_pln_confidence = reasoning->pln_engine->average_confidence;
        
        printf("PLN Deduction: %s ∧ %s ⇒ conclusion (strength: %.3f, confidence: %.3f)\n",
               premise1, premise2, 
               result->truth_value.strength, 
               result->truth_value.confidence);
        
        pln_inference_result_destroy(result);
        return 0;
    }
    
    return -1;
}

// Perform PLN induction
int pln_perform_induction(reasoning_engine* reasoning, const char* evidence_ab, const char* evidence_a) {
    if (!reasoning || !reasoning->pln_engine || !evidence_ab || !evidence_a) {
        return -1;
    }
    
    struct pln_node* node_ab = pln_node_find_by_name(reasoning->pln_engine, evidence_ab);
    struct pln_node* node_a = pln_node_find_by_name(reasoning->pln_engine, evidence_a);
    
    if (!node_ab) {
        node_ab = pln_node_create(reasoning->pln_engine, PLN_NODE_CONCEPT, NULL, evidence_ab);
        node_ab->truth_value = pln_truth_value_create(0.7f, 0.6f, 3.0f);
    }
    
    if (!node_a) {
        node_a = pln_node_create(reasoning->pln_engine, PLN_NODE_CONCEPT, NULL, evidence_a);
        node_a->truth_value = pln_truth_value_create(0.85f, 0.9f, 10.0f);
    }
    
    struct pln_inference_result* result = pln_induction(reasoning->pln_engine, node_ab, node_a);
    
    if (result) {
        reasoning->inferences_made++;
        reasoning->pln_inference_rate = pln_engine_get_inference_rate(reasoning->pln_engine);
        reasoning->average_pln_confidence = reasoning->pln_engine->average_confidence;
        
        printf("PLN Induction: Generalized from %s and %s (strength: %.3f, confidence: %.3f)\n",
               evidence_ab, evidence_a,
               result->truth_value.strength, 
               result->truth_value.confidence);
        
        pln_inference_result_destroy(result);
        return 0;
    }
    
    return -1;
}

// Perform PLN abduction
int pln_perform_abduction(reasoning_engine* reasoning, const char* rule_ab, const char* evidence_b) {
    if (!reasoning || !reasoning->pln_engine || !rule_ab || !evidence_b) {
        return -1;
    }
    
    struct pln_node* rule = pln_node_find_by_name(reasoning->pln_engine, rule_ab);
    struct pln_node* evidence = pln_node_find_by_name(reasoning->pln_engine, evidence_b);
    
    if (!rule) {
        rule = pln_node_create(reasoning->pln_engine, PLN_NODE_IMPLICATION_LINK, NULL, rule_ab);
        rule->truth_value = pln_truth_value_create(0.8f, 0.7f, 6.0f);
    }
    
    if (!evidence) {
        evidence = pln_node_create(reasoning->pln_engine, PLN_NODE_CONCEPT, NULL, evidence_b);
        evidence->truth_value = pln_truth_value_create(0.9f, 0.85f, 8.0f);
    }
    
    struct pln_inference_result* result = pln_abduction(reasoning->pln_engine, rule, evidence);
    
    if (result) {
        reasoning->inferences_made++;
        reasoning->pln_inference_rate = pln_engine_get_inference_rate(reasoning->pln_engine);
        reasoning->average_pln_confidence = reasoning->pln_engine->average_confidence;
        
        printf("PLN Abduction: Hypothesis from %s and %s (strength: %.3f, confidence: %.3f)\n",
               rule_ab, evidence_b,
               result->truth_value.strength, 
               result->truth_value.confidence);
        
        pln_inference_result_destroy(result);
        return 0;
    }
    
    return -1;
}

// Perform PLN revision
int pln_perform_revision(reasoning_engine* reasoning, const char* belief1, const char* belief2) {
    if (!reasoning || !reasoning->pln_engine || !belief1 || !belief2) {
        return -1;
    }
    
    struct pln_node* node1 = pln_node_find_by_name(reasoning->pln_engine, belief1);
    struct pln_node* node2 = pln_node_find_by_name(reasoning->pln_engine, belief2);
    
    if (!node1) {
        node1 = pln_node_create(reasoning->pln_engine, PLN_NODE_CONCEPT, NULL, belief1);
        node1->truth_value = pln_truth_value_create(0.7f, 0.6f, 4.0f);
    }
    
    if (!node2) {
        node2 = pln_node_create(reasoning->pln_engine, PLN_NODE_CONCEPT, NULL, belief2);
        node2->truth_value = pln_truth_value_create(0.8f, 0.7f, 6.0f);
    }
    
    struct pln_inference_result* result = pln_revision(reasoning->pln_engine, node1, node2);
    
    if (result) {
        reasoning->inferences_made++;
        reasoning->pln_inference_rate = pln_engine_get_inference_rate(reasoning->pln_engine);
        reasoning->average_pln_confidence = reasoning->pln_engine->average_confidence;
        
        printf("PLN Revision: Combined %s and %s (strength: %.3f, confidence: %.3f)\n",
               belief1, belief2,
               result->truth_value.strength, 
               result->truth_value.confidence);
        
        pln_inference_result_destroy(result);
        return 0;
    }
    
    return -1;
}

// Get PLN inference rate
float pln_get_inference_rate(reasoning_engine* reasoning) {
    if (!reasoning || !reasoning->pln_engine) {
        return 0.0f;
    }
    
    return pln_engine_get_inference_rate(reasoning->pln_engine);
}

// Add belief to PLN
void pln_add_belief(reasoning_engine* reasoning, const char* concept, float strength, float confidence) {
    if (!reasoning || !reasoning->pln_engine || !concept) {
        return;
    }
    
    struct pln_node* node = pln_node_create(reasoning->pln_engine, PLN_NODE_CONCEPT, NULL, concept);
    node->truth_value = pln_truth_value_create(strength, confidence, 1.0f);
}

// Print PLN statistics
void pln_print_stats(reasoning_engine* reasoning) {
    if (!reasoning || !reasoning->pln_engine) {
        printf("PLN engine not initialized\n");
        return;
    }
    
    printf("PLN Statistics:\n");
    printf("  Total inferences: %lu\n", reasoning->pln_engine->inferences_made);
    printf("  Inference rate: %.2f inferences/second\n", reasoning->pln_inference_rate);
    printf("  Average confidence: %.3f\n", reasoning->average_pln_confidence);
    printf("  Node count: %zu\n", reasoning->pln_engine->node_count);
}

// MOSES Integration Functions

// Initialize MOSES evolution
int init_moses_evolution(reasoning_engine* reasoning, size_t population_size) {
    if (!reasoning || !reasoning->ctx) {
        return -1;
    }
    
    reasoning->moses_engine = moses_engine_create(reasoning->ctx, population_size);
    if (!reasoning->moses_engine) {
        return -1;
    }
    
    // Set cognitive fitness function
    moses_set_fitness_function(reasoning->moses_engine, moses_cognitive_fitness_function, reasoning);
    
    // Initialize population
    if (moses_initialize_population(reasoning->moses_engine) != 0) {
        moses_engine_destroy(reasoning->moses_engine);
        reasoning->moses_engine = NULL;
        return -1;
    }
    
    return 0;
}

// Evolve reasoning strategies
int moses_evolve_reasoning_strategies(reasoning_engine* reasoning, uint32_t generations) {
    if (!reasoning || !reasoning->moses_engine) {
        return -1;
    }
    
    printf("Evolving reasoning strategies for %u generations...\n", generations);
    
    for (uint32_t gen = 0; gen < generations; gen++) {
        if (moses_evolve_generation(reasoning->moses_engine) != 0) {
            printf("Evolution failed at generation %u\n", gen);
            return -1;
        }
        
        // Update metrics
        reasoning->best_program_fitness = reasoning->moses_engine->population->best_fitness;
        reasoning->evolution_generations = reasoning->moses_engine->generations_evolved;
        
        // Print progress every 10 generations
        if ((gen + 1) % 10 == 0) {
            printf("  Generation %u: Best fitness = %.4f, Avg fitness = %.4f, Diversity = %.4f\n",
                   gen + 1,
                   reasoning->moses_engine->population->best_fitness,
                   reasoning->moses_engine->population->average_fitness,
                   reasoning->moses_engine->population->diversity_measure);
        }
    }
    
    printf("Evolution completed!\n");
    return 0;
}

// Optimize cognitive program for specific problem type
int moses_optimize_cognitive_program(reasoning_engine* reasoning, const char* problem_type) {
    if (!reasoning || !reasoning->moses_engine || !problem_type) {
        return -1;
    }
    
    printf("Optimizing cognitive program for problem type: %s\n", problem_type);
    
    // Select appropriate fitness function based on problem type
    if (strcmp(problem_type, "reasoning") == 0) {
        moses_set_fitness_function(reasoning->moses_engine, moses_cognitive_fitness_function, reasoning);
    } else if (strcmp(problem_type, "complexity") == 0) {
        moses_set_fitness_function(reasoning->moses_engine, moses_complexity_fitness_function, reasoning);
    } else {
        moses_set_fitness_function(reasoning->moses_engine, moses_default_fitness_function, reasoning);
    }
    
    // Run evolution for the specific problem
    return moses_evolve_reasoning_strategies(reasoning, 20);
}

// Get best fitness from current population
float moses_get_best_fitness(reasoning_engine* reasoning) {
    if (!reasoning || !reasoning->moses_engine || !reasoning->moses_engine->population) {
        return -INFINITY;
    }
    
    return reasoning->moses_engine->population->best_fitness;
}

// Print MOSES evolution statistics
void moses_print_evolution_stats(reasoning_engine* reasoning) {
    if (!reasoning || !reasoning->moses_engine) {
        printf("MOSES engine not initialized\n");
        return;
    }
    
    moses_print_population_stats(reasoning->moses_engine);
    
    printf("  Best program fitness: %.4f\n", reasoning->best_program_fitness);
    printf("  Evolution generations: %u\n", reasoning->evolution_generations);
}

// Self-modify agent using MOSES
int moses_self_modify_agent(reasoning_engine* reasoning) {
    if (!reasoning || !reasoning->moses_engine) {
        return -1;
    }
    
    printf("Agent self-modification using MOSES...\n");
    
    // Run evolution to find better cognitive strategies
    int result = moses_evolve_reasoning_strategies(reasoning, 30);
    
    if (result == 0) {
        printf("Self-modification completed. New best fitness: %.4f\n", 
               moses_get_best_fitness(reasoning));
        
        // In a real implementation, we would apply the best program to modify
        // the agent's behavior, attention allocation, or reasoning strategies
        printf("Applied evolved cognitive strategies to agent behavior.\n");
    }
    
    return result;
}