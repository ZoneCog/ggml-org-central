#pragma once

#include "ggml.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations for cognitive agent integration
struct cognitive_agent;
struct attention_economy;
struct reasoning_engine;

// Self-modification operation types
enum self_mod_type {
    SELF_MOD_ATTENTION_WEIGHTS = 1,     // Modify attention allocation parameters
    SELF_MOD_MEMORY_STRUCTURE = 2,      // Restructure memory organization
    SELF_MOD_REASONING_RULES = 3,       // Add/modify inference rules
    SELF_MOD_COMMUNICATION_PATTERNS = 4, // Modify inter-agent communication
    SELF_MOD_GOAL_HIERARCHY = 5,        // Modify goal prioritization
    SELF_MOD_LEARNING_RATE = 6          // Adjust learning parameters
};

// Modification safety levels
enum safety_level {
    SAFETY_LEVEL_SAFE = 1,              // Guaranteed reversible
    SAFETY_LEVEL_MONITORED = 2,         // Monitored with rollback
    SAFETY_LEVEL_EXPERIMENTAL = 3,      // High risk, careful monitoring
    SAFETY_LEVEL_PROHIBITED = 4         // Not allowed
};

// Self-modification record for rollback
typedef struct self_mod_record {
    uint64_t modification_id;           // Unique modification ID
    enum self_mod_type mod_type;        // Type of modification
    enum safety_level safety_level;    // Safety assessment
    uint64_t timestamp;                 // When modification was made
    float* backup_data;                 // Backup of original state
    size_t backup_size;                 // Size of backup data
    float performance_before;           // Performance before modification
    float performance_after;            // Performance after modification
    uint32_t rollback_count;            // Number of rollbacks performed
    bool is_active;                     // Whether modification is active
} self_mod_record;

// Self-modification sandbox for safe execution
typedef struct self_mod_sandbox {
    struct ggml_context* sandbox_ctx;   // Isolated computation context
    struct cognitive_agent* test_agent; // Agent copy for testing
    
    // Safety constraints
    float max_performance_degradation;  // Maximum allowed performance loss
    uint32_t max_test_cycles;          // Maximum test cycles before rollback
    float attention_budget_limit;       // Maximum attention that can be modified
    
    // Monitoring state
    bool is_active;                     // Whether sandbox is running
    uint64_t test_start_time;          // When current test started
    uint32_t current_test_cycle;       // Current test cycle
    float baseline_performance;        // Performance before modification
} self_mod_sandbox;

// Meta-learning experience record
typedef struct meta_learning_experience {
    enum self_mod_type modification_type; // Type of modification attempted
    float context_vector[64];            // Context when modification was made
    float performance_delta;             // Change in performance
    float attention_cost;               // Attention cost of modification
    uint64_t timestamp;                 // When experience was recorded
    bool was_successful;                // Whether modification was beneficial
} meta_learning_experience;

// Meta-learning system for learning-to-learn
typedef struct meta_learning_system {
    struct ggml_context* ctx;
    
    // Experience replay buffer
    meta_learning_experience* experiences;
    size_t experience_count;
    size_t experience_capacity;
    
    // Learning parameters
    float learning_rate;                // Meta-learning rate
    float exploration_rate;             // Exploration vs exploitation
    float adaptation_threshold;         // Threshold for triggering adaptation
    
    // Strategy selection
    struct ggml_tensor* strategy_weights; // Weights for different strategies
    float strategy_success_rates[10];    // Success rates for strategies
    uint32_t strategy_usage_counts[10];  // Usage counts for strategies
    
    // Performance tracking
    float adaptation_speed;             // How quickly agent adapts
    float meta_learning_efficiency;     // Efficiency of meta-learning
    uint64_t adaptations_performed;     // Total adaptations performed
} meta_learning_system;

// Main self-modification engine
typedef struct self_modification_engine {
    struct ggml_context* ctx;
    
    // Core components
    self_mod_sandbox* sandbox;          // Safe execution environment
    meta_learning_system* meta_learning; // Meta-learning capabilities
    
    // Modification history
    self_mod_record* modification_history;
    size_t history_count;
    size_t history_capacity;
    
    // Safety constraints
    float safety_threshold;             // Overall safety threshold
    uint32_t max_concurrent_mods;       // Max concurrent modifications
    uint32_t rollback_limit;           // Max rollbacks before disabling
    
    // Performance monitoring
    float performance_baseline;         // Baseline performance measure
    float current_performance;          // Current performance measure
    uint64_t total_modifications;       // Total modifications attempted
    uint64_t successful_modifications;  // Successful modifications
    uint64_t rollbacks_performed;       // Total rollbacks performed
    
    // Meta-cognitive monitoring
    float self_awareness_level;         // Level of self-awareness
    float modification_confidence;      // Confidence in modifications
    float learning_progress;           // Progress in meta-learning
} self_modification_engine;

// Core engine functions
self_modification_engine* init_self_modification_engine(struct ggml_context* ctx);
void cleanup_self_modification_engine(self_modification_engine* engine);

// Self-modification functions
bool propose_self_modification(self_modification_engine* engine, 
                              enum self_mod_type mod_type,
                              const void* modification_params,
                              size_t params_size);
bool execute_self_modification(self_modification_engine* engine,
                              struct cognitive_agent* agent,
                              uint64_t modification_id);
bool rollback_modification(self_modification_engine* engine,
                          struct cognitive_agent* agent,
                          uint64_t modification_id);

// Safety assessment functions
enum safety_level assess_modification_safety(self_modification_engine* engine,
                                            enum self_mod_type mod_type,
                                            const void* params);
float estimate_modification_impact(self_modification_engine* engine,
                                  enum self_mod_type mod_type,
                                  const void* params);
bool validate_safety_constraints(self_modification_engine* engine,
                                struct cognitive_agent* agent);

// Sandbox functions
self_mod_sandbox* init_sandbox(struct ggml_context* ctx);
void cleanup_sandbox(self_mod_sandbox* sandbox);
bool test_modification_in_sandbox(self_mod_sandbox* sandbox,
                                 enum self_mod_type mod_type,
                                 const void* params,
                                 float* performance_result);

// Meta-learning functions
meta_learning_system* init_meta_learning_system(struct ggml_context* ctx);
void cleanup_meta_learning_system(meta_learning_system* meta_learning);
void record_learning_experience(meta_learning_system* meta_learning,
                               enum self_mod_type mod_type,
                               const float* context,
                               float performance_delta,
                               bool success);
enum self_mod_type select_adaptation_strategy(meta_learning_system* meta_learning,
                                             const float* current_context);
void update_strategy_weights(meta_learning_system* meta_learning,
                           enum self_mod_type strategy,
                           float performance_delta);

// Performance monitoring functions
void update_performance_metrics(self_modification_engine* engine,
                               struct cognitive_agent* agent);
float measure_cognitive_performance(struct cognitive_agent* agent);
float compute_adaptation_efficiency(self_modification_engine* engine);
void log_modification_attempt(self_modification_engine* engine,
                             enum self_mod_type mod_type,
                             bool success,
                             float performance_change);

// Meta-cognitive monitoring functions
void update_self_awareness_metrics(self_modification_engine* engine,
                                  struct cognitive_agent* agent);
float assess_modification_confidence(self_modification_engine* engine,
                                    enum self_mod_type mod_type);
void monitor_learning_progress(self_modification_engine* engine);

// Integration with cognitive agent
bool integrate_self_modification(struct cognitive_agent* agent,
                               self_modification_engine* engine);
void process_self_modification_cycle(struct cognitive_agent* agent);

#ifdef __cplusplus
}
#endif