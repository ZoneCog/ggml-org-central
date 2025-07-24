#pragma once

#include "ggml.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// PLN Truth Value structure as specified in the issue
struct pln_truth_value {
    float strength;     // Probability/frequency
    float confidence;   // Sample size confidence  
    float count;        // Evidence count
};

// PLN Node types
enum pln_node_type {
    PLN_NODE_CONCEPT = 1,
    PLN_NODE_LINK = 2,
    PLN_NODE_IMPLICATION_LINK = 3,
    PLN_NODE_INHERITANCE_LINK = 4,
    PLN_NODE_SIMILARITY_LINK = 5,
    PLN_NODE_AND_LINK = 6,
    PLN_NODE_OR_LINK = 7
};

// PLN Node structure as specified in the issue
struct pln_node {
    struct ggml_tensor* content;        // Node content as tensor
    struct pln_truth_value truth_value; // PLN truth value
    uint32_t node_type;                 // ConceptNode, LinkNode, etc.
    struct pln_node** incoming;         // Incoming links
    struct pln_node** outgoing;         // Outgoing links
    size_t incoming_count;              // Number of incoming links
    size_t outgoing_count;              // Number of outgoing links
    size_t incoming_capacity;           // Capacity for incoming links
    size_t outgoing_capacity;           // Capacity for outgoing links
    uint64_t node_id;                   // Unique identifier
    char* name;                         // Optional name for debugging
};

// PLN Inference Rule types
enum pln_rule_type {
    PLN_RULE_DEDUCTION = 1,
    PLN_RULE_INDUCTION = 2,
    PLN_RULE_ABDUCTION = 3,
    PLN_RULE_REVISION = 4,
    PLN_RULE_MODUS_PONENS = 5
};

// PLN Inference Result
struct pln_inference_result {
    struct pln_node* conclusion;
    struct pln_truth_value truth_value;
    enum pln_rule_type rule_used;
    float confidence_level;
    struct pln_node** premises;
    size_t premise_count;
};

// PLN Engine structure
struct pln_engine {
    struct ggml_context* ctx;
    struct pln_node** nodes;
    size_t node_count;
    size_t capacity;
    uint64_t next_node_id;
    
    // Performance metrics
    uint64_t inferences_made;
    uint64_t total_inference_time_ns;
    float average_confidence;
};

// PLN Core Functions
struct pln_engine* pln_engine_create(struct ggml_context* ctx);
void pln_engine_destroy(struct pln_engine* engine);

// Truth Value Operations
struct pln_truth_value pln_truth_value_create(float strength, float confidence, float count);
struct pln_truth_value pln_truth_value_deduction(
    const struct pln_truth_value* tv_ab, 
    const struct pln_truth_value* tv_bc
);
struct pln_truth_value pln_truth_value_induction(
    const struct pln_truth_value* tv_ab, 
    const struct pln_truth_value* tv_a
);
struct pln_truth_value pln_truth_value_revision(
    const struct pln_truth_value* tv1, 
    const struct pln_truth_value* tv2
);
float pln_truth_value_expectation(const struct pln_truth_value* tv);

// Node Operations
struct pln_node* pln_node_create(
    struct pln_engine* engine,
    enum pln_node_type type,
    struct ggml_tensor* content,
    const char* name
);
void pln_node_destroy(struct pln_node* node);
void pln_node_add_outgoing(struct pln_node* from, struct pln_node* to);
void pln_node_add_incoming(struct pln_node* to, struct pln_node* from);
struct pln_node* pln_node_find_by_name(struct pln_engine* engine, const char* name);

// Inference Rules Implementation
struct pln_inference_result* pln_deduction(
    struct pln_engine* engine,
    struct pln_node* premise_ab,  // A->B
    struct pln_node* premise_bc   // B->C
);

struct pln_inference_result* pln_induction(
    struct pln_engine* engine,
    struct pln_node* evidence_ab, // A->B evidence
    struct pln_node* evidence_a   // A evidence
);

struct pln_inference_result* pln_abduction(
    struct pln_engine* engine,
    struct pln_node* rule_ab,     // A->B rule
    struct pln_node* evidence_b   // B evidence
);

struct pln_inference_result* pln_revision(
    struct pln_engine* engine,
    struct pln_node* belief1,     // First belief
    struct pln_node* belief2      // Second belief about same concept
);

void pln_inference_result_destroy(struct pln_inference_result* result);

// Integration with existing hypergraph memory
struct pln_node* pln_from_hypergraph_node(
    struct pln_engine* engine,
    void* hypergraph_node  // hypergraph_node from cognitive-agent.h
);

// Performance and Statistics
float pln_engine_get_inference_rate(struct pln_engine* engine);
void pln_engine_reset_stats(struct pln_engine* engine);

// Utility functions
uint64_t pln_get_timestamp_ns(void);
bool pln_truth_value_is_valid(const struct pln_truth_value* tv);

#ifdef __cplusplus
}
#endif