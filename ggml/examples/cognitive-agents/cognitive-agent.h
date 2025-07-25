#pragma once

#include "ggml.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration for PLN integration
struct pln_engine;

// Forward declaration for MOSES integration  
struct moses_engine;

// Forward declaration for Pattern Matcher integration
struct pattern_matcher;

// Forward declaration for Self-Modification integration
struct self_modification_engine;

// Forward declaration for Behavior Analysis integration
struct behavior_analysis_engine;

// Forward declarations
typedef struct cognitive_agent cognitive_agent;
typedef struct hypergraph_memory hypergraph_memory;
typedef struct task_orchestrator task_orchestrator;
typedef struct reasoning_engine reasoning_engine;
typedef struct attention_economy attention_economy;

// Simplified cognitive tensor packet (without RPC dependency)
typedef struct {
    // Cognitive metadata only (tensor data handled separately)
    float attention_weight;          // Economic attention value
    uint32_t cognitive_type;         // Type of cognitive operation
    uint64_t source_agent_id;        // Originating agent
    uint64_t target_agent_id;        // Target agent
    char meta_context[256];          // Context information
    float salience_score;            // Relevance measure
    uint32_t recursion_depth;        // Self-reference depth
    uint64_t timestamp;              // When created
} cognitive_tensor_packet;

// Cognitive operation types
enum cognitive_type {
    COGNITIVE_TYPE_MEMORY = 1,
    COGNITIVE_TYPE_TASK = 2,
    COGNITIVE_TYPE_REASONING = 3,
    COGNITIVE_TYPE_ATTENTION = 4,
    COGNITIVE_TYPE_COMMUNICATION = 5
};

// Hypergraph node types
enum node_type {
    NODE_TYPE_CONCEPT = 1,
    NODE_TYPE_LINK = 2,
    NODE_TYPE_RELATION = 3
};

// Hypergraph node structure
typedef struct hypergraph_node {
    struct ggml_tensor* data;           // Node content as tensor
    struct ggml_tensor* embedding;      // Semantic embedding
    uint32_t node_type;          // Node type
    float truth_value;           // PLN truth value
    float confidence;            // PLN confidence
    uint64_t creation_time;      // Timestamp
    uint64_t last_access;        // Last accessed
} hypergraph_node;

// Hypergraph memory system
struct hypergraph_memory {
    struct ggml_context* ctx;
    
    // Storage
    hypergraph_node** nodes;
    size_t node_count;
    size_t capacity;
    
    // Statistics
    uint64_t total_accesses;
    uint64_t cache_hits;
};

// Attention economy state
struct attention_economy {
    float total_attention;       // Total available attention
    float allocated_attention;   // Currently allocated
    
    // Attention allocation per cognitive function
    float memory_allocation;
    float reasoning_allocation;
    float communication_allocation;
    float self_modification_allocation;
    
    // Economic parameters
    float min_threshold;         // Minimum attention to process
    float decay_rate;           // Attention decay over time
    float novelty_bonus;        // Bonus for novel information
    
    // Performance tracking
    float performance_history[100];
    size_t history_index;
    size_t history_size;
};

// Task orchestrator
struct task_orchestrator {
    // Current task state
    struct ggml_tensor* current_goal;
    struct ggml_tensor* context_state;
    
    // Task queue
    void** pending_tasks;
    size_t task_count;
    size_t task_capacity;
};

// Reasoning engine
struct reasoning_engine {
    struct ggml_context* ctx;
    
    // Traditional reasoning state
    struct ggml_tensor* current_beliefs;
    struct ggml_tensor* inference_rules;
    
    // PLN Integration
    struct pln_engine* pln_engine;
    
    // MOSES Integration
    struct moses_engine* moses_engine;
    
    // Pattern Matcher Integration
    struct pattern_matcher* pattern_matcher;
    
    // Performance metrics
    float reasoning_accuracy;
    uint64_t inferences_made;
    
    // PLN specific metrics
    float pln_inference_rate;
    float average_pln_confidence;
    
    // MOSES specific metrics
    float best_program_fitness;
    uint32_t evolution_generations;
    
    // Pattern matching metrics
    float pattern_match_accuracy;
    uint64_t patterns_recognized;
};

// Main cognitive agent structure
struct cognitive_agent {
    struct ggml_context* ctx;
    struct ggml_backend* backend;
    
    // Cognitive components
    hypergraph_memory* memory;
    task_orchestrator* tasks;
    reasoning_engine* reasoning;
    attention_economy* attention;
    struct self_modification_engine* self_modification;
    struct behavior_analysis_engine* behavior_analysis;
    
    // Network identity
    uint64_t agent_id;
    char endpoint[256];
    float attention_weight;
    
    // State tracking
    uint64_t messages_sent;
    uint64_t messages_received;
    uint64_t cycles_completed;
};

// Core agent functions
cognitive_agent* create_cognitive_agent(const char* endpoint);
void cleanup_cognitive_agent(cognitive_agent* agent);
uint64_t generate_agent_id(void);

// Memory system functions
hypergraph_memory* init_hypergraph_memory(struct ggml_context* ctx);
void cleanup_hypergraph_memory(hypergraph_memory* mem);
void add_knowledge(hypergraph_memory* mem, const char* concept, 
                   float* embedding, size_t emb_size);
hypergraph_node* find_concept(hypergraph_memory* mem, const char* concept);

// Attention economy functions
attention_economy* init_attention_economy(void);
void cleanup_attention_economy(attention_economy* attention);
float compute_salience(struct ggml_tensor* tensor, attention_economy* attention);
void allocate_attention(attention_economy* attention, float amount, uint32_t target);
void update_performance_history(attention_economy* attention, float performance);

// Task orchestration functions
task_orchestrator* init_task_orchestrator(void);
void cleanup_task_orchestrator(task_orchestrator* orch);

// Reasoning engine functions
reasoning_engine* init_reasoning_engine(struct ggml_context* ctx);
void cleanup_reasoning_engine(reasoning_engine* reasoning);

// PLN Integration functions
int init_pln_reasoning(reasoning_engine* reasoning);
int pln_perform_deduction(reasoning_engine* reasoning, const char* premise1, const char* premise2);
int pln_perform_induction(reasoning_engine* reasoning, const char* evidence_ab, const char* evidence_a);
int pln_perform_abduction(reasoning_engine* reasoning, const char* rule_ab, const char* evidence_b);
int pln_perform_revision(reasoning_engine* reasoning, const char* belief1, const char* belief2);
float pln_get_inference_rate(reasoning_engine* reasoning);
void pln_add_belief(reasoning_engine* reasoning, const char* concept, float strength, float confidence);
void pln_print_stats(reasoning_engine* reasoning);

// MOSES Integration functions
int init_moses_evolution(reasoning_engine* reasoning, size_t population_size);
int moses_evolve_reasoning_strategies(reasoning_engine* reasoning, uint32_t generations);
int moses_optimize_cognitive_program(reasoning_engine* reasoning, const char* problem_type);
float moses_get_best_fitness(reasoning_engine* reasoning);
void moses_print_evolution_stats(reasoning_engine* reasoning);
int moses_self_modify_agent(reasoning_engine* reasoning);

// Pattern Matching Integration functions
int init_pattern_matching(reasoning_engine* reasoning);
int pattern_add_knowledge_pattern(reasoning_engine* reasoning, const char* concept, struct ggml_tensor* data);
int pattern_recognize_sequence(reasoning_engine* reasoning, struct ggml_tensor* sequence_data);
int pattern_find_analogies(reasoning_engine* reasoning, const char* source_concept, const char* target_concept);
float pattern_get_match_accuracy(reasoning_engine* reasoning);
void pattern_print_recognition_stats(reasoning_engine* reasoning);
int pattern_cross_modal_analysis(reasoning_engine* reasoning, struct ggml_tensor* text, struct ggml_tensor* embedding);

// Communication functions
void send_cognitive_tensor(cognitive_agent* sender, uint64_t target_agent_id,
                          struct ggml_tensor* tensor, float attention_weight);
void process_incoming_tensor(cognitive_agent* receiver, 
                           cognitive_tensor_packet* msg);

// Utility functions
uint64_t get_timestamp(void);
float compute_tensor_similarity(struct ggml_tensor* a, struct ggml_tensor* b);
uint32_t infer_cognitive_type(struct ggml_tensor* tensor);

#ifdef __cplusplus
}
#endif