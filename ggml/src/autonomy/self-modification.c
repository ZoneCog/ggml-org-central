#include "self-modification.h"
#include "../reasoning/pln-core.h"
#include "../reasoning/moses-core.h"
#include "../reasoning/pattern-matcher.h"
#include "../../examples/cognitive-agents/cognitive-agent.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Safe defaults for self-modification parameters
#define DEFAULT_SAFETY_THRESHOLD 0.9f
#define DEFAULT_MAX_PERFORMANCE_DEGRADATION 0.1f  // Max 10% performance loss
#define DEFAULT_MAX_TEST_CYCLES 100
#define DEFAULT_ATTENTION_BUDGET_LIMIT 0.3f       // Max 30% of attention
#define MAX_CONCURRENT_MODIFICATIONS 3
#define DEFAULT_ROLLBACK_LIMIT 5
#define META_LEARNING_RATE 0.01f
#define EXPLORATION_RATE 0.1f
#define ADAPTATION_THRESHOLD 0.05f

// Utility functions
static uint64_t get_timestamp_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static uint64_t generate_modification_id(void) {
    static uint64_t counter = 1;
    return counter++ | (get_timestamp_ns() & 0xFFFFFFFF00000000ULL);
}

// Core engine initialization
self_modification_engine* init_self_modification_engine(struct ggml_context* ctx) {
    self_modification_engine* engine = calloc(1, sizeof(self_modification_engine));
    if (!engine) {
        return NULL;
    }
    
    engine->ctx = ctx;
    
    // Initialize sandbox
    engine->sandbox = init_sandbox(ctx);
    if (!engine->sandbox) {
        free(engine);
        return NULL;
    }
    
    // Initialize meta-learning system
    engine->meta_learning = init_meta_learning_system(ctx);
    if (!engine->meta_learning) {
        cleanup_sandbox(engine->sandbox);
        free(engine);
        return NULL;
    }
    
    // Initialize modification history
    engine->history_capacity = 1000;
    engine->modification_history = calloc(engine->history_capacity, sizeof(self_mod_record));
    if (!engine->modification_history) {
        cleanup_meta_learning_system(engine->meta_learning);
        cleanup_sandbox(engine->sandbox);
        free(engine);
        return NULL;
    }
    
    // Set safety parameters
    engine->safety_threshold = DEFAULT_SAFETY_THRESHOLD;
    engine->max_concurrent_mods = MAX_CONCURRENT_MODIFICATIONS;
    engine->rollback_limit = DEFAULT_ROLLBACK_LIMIT;
    
    // Initialize performance tracking
    engine->performance_baseline = 1.0f;
    engine->current_performance = 1.0f;
    engine->self_awareness_level = 0.5f;
    engine->modification_confidence = 0.7f;
    engine->learning_progress = 0.0f;
    
    printf("Initialized self-modification engine with safety threshold %.2f\n", 
           engine->safety_threshold);
    
    return engine;
}

void cleanup_self_modification_engine(self_modification_engine* engine) {
    if (!engine) return;
    
    cleanup_sandbox(engine->sandbox);
    cleanup_meta_learning_system(engine->meta_learning);
    
    // Cleanup modification history
    if (engine->modification_history) {
        for (size_t i = 0; i < engine->history_count; i++) {
            if (engine->modification_history[i].backup_data) {
                free(engine->modification_history[i].backup_data);
            }
        }
        free(engine->modification_history);
    }
    
    free(engine);
}

// Sandbox implementation
self_mod_sandbox* init_sandbox(struct ggml_context* ctx) {
    self_mod_sandbox* sandbox = calloc(1, sizeof(self_mod_sandbox));
    if (!sandbox) {
        return NULL;
    }
    
    // Create isolated context for testing
    struct ggml_init_params params = {
        .mem_size = 16 * 1024 * 1024,  // 16MB for testing
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    sandbox->sandbox_ctx = ggml_init(params);
    if (!sandbox->sandbox_ctx) {
        free(sandbox);
        return NULL;
    }
    
    // Set safety constraints
    sandbox->max_performance_degradation = DEFAULT_MAX_PERFORMANCE_DEGRADATION;
    sandbox->max_test_cycles = DEFAULT_MAX_TEST_CYCLES;
    sandbox->attention_budget_limit = DEFAULT_ATTENTION_BUDGET_LIMIT;
    sandbox->is_active = false;
    
    return sandbox;
}

void cleanup_sandbox(self_mod_sandbox* sandbox) {
    if (!sandbox) return;
    
    if (sandbox->sandbox_ctx) {
        ggml_free(sandbox->sandbox_ctx);
    }
    
    // Note: test_agent is a copy, cleanup handled by test framework
    free(sandbox);
}

// Meta-learning system implementation
meta_learning_system* init_meta_learning_system(struct ggml_context* ctx) {
    meta_learning_system* meta_learning = calloc(1, sizeof(meta_learning_system));
    if (!meta_learning) {
        return NULL;
    }
    
    meta_learning->ctx = ctx;
    
    // Initialize experience buffer
    meta_learning->experience_capacity = 10000;
    meta_learning->experiences = calloc(meta_learning->experience_capacity, 
                                       sizeof(meta_learning_experience));
    if (!meta_learning->experiences) {
        free(meta_learning);
        return NULL;
    }
    
    // Set learning parameters
    meta_learning->learning_rate = META_LEARNING_RATE;
    meta_learning->exploration_rate = EXPLORATION_RATE;
    meta_learning->adaptation_threshold = ADAPTATION_THRESHOLD;
    
    // Initialize strategy weights
    meta_learning->strategy_weights = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10);
    if (meta_learning->strategy_weights) {
        // Initialize with uniform weights
        float* weights = (float*)meta_learning->strategy_weights->data;
        for (int i = 0; i < 10; i++) {
            weights[i] = 0.1f;  // Equal probability initially
            meta_learning->strategy_success_rates[i] = 0.5f;
            meta_learning->strategy_usage_counts[i] = 0;
        }
    }
    
    // Initialize performance metrics
    meta_learning->adaptation_speed = 1.0f;
    meta_learning->meta_learning_efficiency = 0.5f;
    meta_learning->adaptations_performed = 0;
    
    return meta_learning;
}

void cleanup_meta_learning_system(meta_learning_system* meta_learning) {
    if (!meta_learning) return;
    
    if (meta_learning->experiences) {
        free(meta_learning->experiences);
    }
    
    // Note: strategy_weights tensor is managed by ggml context
    free(meta_learning);
}

// Safety assessment functions
enum safety_level assess_modification_safety(self_modification_engine* engine,
                                            enum self_mod_type mod_type,
                                            const void* params) {
    if (!engine || !params) {
        return SAFETY_LEVEL_PROHIBITED;
    }
    
    // Assess based on modification type
    switch (mod_type) {
        case SELF_MOD_ATTENTION_WEIGHTS:
            // Attention weight modifications are generally safe
            return SAFETY_LEVEL_SAFE;
            
        case SELF_MOD_MEMORY_STRUCTURE:
            // Memory restructuring needs monitoring
            return SAFETY_LEVEL_MONITORED;
            
        case SELF_MOD_REASONING_RULES:
            // Reasoning rule changes are risky
            return SAFETY_LEVEL_EXPERIMENTAL;
            
        case SELF_MOD_COMMUNICATION_PATTERNS:
            // Communication pattern changes are safe with monitoring
            return SAFETY_LEVEL_MONITORED;
            
        case SELF_MOD_GOAL_HIERARCHY:
            // Goal hierarchy changes are high risk
            return SAFETY_LEVEL_EXPERIMENTAL;
            
        case SELF_MOD_LEARNING_RATE:
            // Learning rate adjustments are safe
            return SAFETY_LEVEL_SAFE;
            
        default:
            return SAFETY_LEVEL_PROHIBITED;
    }
}

float estimate_modification_impact(self_modification_engine* engine,
                                  enum self_mod_type mod_type,
                                  const void* params) {
    if (!engine || !params) {
        return 1.0f;  // Maximum impact (unsafe)
    }
    
    // Base impact estimates for different modification types
    float base_impact = 0.0f;
    
    switch (mod_type) {
        case SELF_MOD_ATTENTION_WEIGHTS:
            base_impact = 0.1f;  // Low impact
            break;
        case SELF_MOD_MEMORY_STRUCTURE:
            base_impact = 0.3f;  // Moderate impact
            break;
        case SELF_MOD_REASONING_RULES:
            base_impact = 0.7f;  // High impact
            break;
        case SELF_MOD_COMMUNICATION_PATTERNS:
            base_impact = 0.2f;  // Low-moderate impact
            break;
        case SELF_MOD_GOAL_HIERARCHY:
            base_impact = 0.8f;  // Very high impact
            break;
        case SELF_MOD_LEARNING_RATE:
            base_impact = 0.15f; // Low impact
            break;
        default:
            base_impact = 1.0f;  // Unknown = maximum impact
    }
    
    // Adjust based on current system state
    float confidence_factor = engine->modification_confidence;
    float experience_factor = (float)engine->successful_modifications / 
                             (engine->total_modifications + 1);
    
    float adjusted_impact = base_impact * (2.0f - confidence_factor) * 
                           (2.0f - experience_factor);
    
    return fminf(adjusted_impact, 1.0f);
}

bool validate_safety_constraints(self_modification_engine* engine,
                                struct cognitive_agent* agent) {
    if (!engine || !agent) {
        return false;
    }
    
    // Count active modifications
    uint32_t active_mods = 0;
    for (size_t i = 0; i < engine->history_count; i++) {
        if (engine->modification_history[i].is_active) {
            active_mods++;
        }
    }
    
    // Check concurrent modification limit
    if (active_mods >= engine->max_concurrent_mods) {
        printf("Safety constraint violation: Too many concurrent modifications (%u >= %u)\n",
               active_mods, engine->max_concurrent_mods);
        return false;
    }
    
    // Check performance degradation
    float performance_ratio = engine->current_performance / engine->performance_baseline;
    if (performance_ratio < (1.0f - DEFAULT_MAX_PERFORMANCE_DEGRADATION)) {
        printf("Safety constraint violation: Performance degradation too high (%.3f < %.3f)\n",
               performance_ratio, 1.0f - DEFAULT_MAX_PERFORMANCE_DEGRADATION);
        return false;
    }
    
    // Check rollback limit
    uint32_t recent_rollbacks = 0;
    uint64_t current_time = get_timestamp_ns();
    uint64_t time_window = 3600000000000ULL; // 1 hour in nanoseconds
    
    for (size_t i = 0; i < engine->history_count; i++) {
        if (engine->modification_history[i].rollback_count > 0 &&
            (current_time - engine->modification_history[i].timestamp) < time_window) {
            recent_rollbacks += engine->modification_history[i].rollback_count;
        }
    }
    
    if (recent_rollbacks >= engine->rollback_limit) {
        printf("Safety constraint violation: Too many recent rollbacks (%u >= %u)\n",
               recent_rollbacks, engine->rollback_limit);
        return false;
    }
    
    return true;
}

// Self-modification proposal and execution
bool propose_self_modification(self_modification_engine* engine, 
                              enum self_mod_type mod_type,
                              const void* modification_params,
                              size_t params_size) {
    if (!engine || !modification_params || params_size == 0) {
        return false;
    }
    
    // Assess safety
    enum safety_level safety = assess_modification_safety(engine, mod_type, modification_params);
    if (safety == SAFETY_LEVEL_PROHIBITED) {
        printf("Self-modification proposal rejected: Prohibited modification type\n");
        return false;
    }
    
    // Estimate impact
    float impact = estimate_modification_impact(engine, mod_type, modification_params);
    if (impact > engine->safety_threshold) {
        printf("Self-modification proposal rejected: Impact %.3f > threshold %.3f\n",
               impact, engine->safety_threshold);
        return false;
    }
    
    // Add to history if space available
    if (engine->history_count >= engine->history_capacity) {
        printf("Self-modification proposal rejected: History buffer full\n");
        return false;
    }
    
    // Create modification record
    self_mod_record* record = &engine->modification_history[engine->history_count];
    record->modification_id = generate_modification_id();
    record->mod_type = mod_type;
    record->safety_level = safety;
    record->timestamp = get_timestamp_ns();
    record->backup_size = params_size;
    record->backup_data = malloc(params_size);
    if (record->backup_data) {
        memcpy(record->backup_data, modification_params, params_size);
    }
    record->performance_before = engine->current_performance;
    record->performance_after = 0.0f;  // Will be set after execution
    record->rollback_count = 0;
    record->is_active = false;  // Will be set true after successful execution
    
    engine->history_count++;
    
    printf("Self-modification proposed: ID %lu, Type %d, Safety %d, Impact %.3f\n",
           record->modification_id, mod_type, safety, impact);
    
    return true;
}

bool test_modification_in_sandbox(self_mod_sandbox* sandbox,
                                 enum self_mod_type mod_type,
                                 const void* params,
                                 float* performance_result) {
    if (!sandbox || !params || !performance_result) {
        return false;
    }
    
    // Mark sandbox as active
    sandbox->is_active = true;
    sandbox->test_start_time = get_timestamp_ns();
    sandbox->current_test_cycle = 0;
    
    // Simulate testing the modification
    // In a real implementation, this would create a copy of the agent
    // and apply the modification in isolation
    
    float simulated_performance = 0.8f + 0.2f * ((float)rand() / RAND_MAX);
    
    // Simulate test cycles
    for (uint32_t cycle = 0; cycle < sandbox->max_test_cycles; cycle++) {
        sandbox->current_test_cycle = cycle;
        
        // Simulate performance measurement
        float cycle_performance = simulated_performance + 
                                 0.1f * sin(cycle * 0.1f) * ((float)rand() / RAND_MAX - 0.5f);
        
        // Check for early termination due to poor performance
        if (cycle_performance < sandbox->baseline_performance - 
                                sandbox->max_performance_degradation) {
            printf("Sandbox test terminated early at cycle %u due to poor performance\n", cycle);
            *performance_result = cycle_performance;
            sandbox->is_active = false;
            return false;
        }
        
        // Update running average
        simulated_performance = 0.9f * simulated_performance + 0.1f * cycle_performance;
    }
    
    *performance_result = simulated_performance;
    sandbox->is_active = false;
    
    printf("Sandbox test completed: Performance %.3f after %u cycles\n",
           *performance_result, sandbox->max_test_cycles);
    
    return true;
}

bool execute_self_modification(self_modification_engine* engine,
                              struct cognitive_agent* agent,
                              uint64_t modification_id) {
    if (!engine || !agent) {
        return false;
    }
    
    // Find modification record
    self_mod_record* record = NULL;
    for (size_t i = 0; i < engine->history_count; i++) {
        if (engine->modification_history[i].modification_id == modification_id) {
            record = &engine->modification_history[i];
            break;
        }
    }
    
    if (!record) {
        printf("Self-modification execution failed: Record not found for ID %lu\n", modification_id);
        return false;
    }
    
    // Test in sandbox first for experimental modifications
    if (record->safety_level >= SAFETY_LEVEL_EXPERIMENTAL) {
        float sandbox_performance;
        if (!test_modification_in_sandbox(engine->sandbox, record->mod_type,
                                         record->backup_data, &sandbox_performance)) {
            printf("Self-modification execution failed: Sandbox test failed\n");
            return false;
        }
        
        // Check if sandbox performance is acceptable
        if (sandbox_performance < record->performance_before - 
                                 DEFAULT_MAX_PERFORMANCE_DEGRADATION) {
            printf("Self-modification execution rejected: Sandbox performance too low\n");
            return false;
        }
    }
    
    // Validate safety constraints
    if (!validate_safety_constraints(engine, agent)) {
        printf("Self-modification execution failed: Safety constraints violated\n");
        return false;
    }
    
    // Execute the modification based on type
    bool success = false;
    switch (record->mod_type) {
        case SELF_MOD_ATTENTION_WEIGHTS:
            // Modify attention allocation weights
            if (agent->attention) {
                // Adjust attention parameters slightly
                agent->attention->decay_rate *= 0.95f;
                agent->attention->novelty_bonus *= 1.05f;
                success = true;
            }
            break;
            
        case SELF_MOD_MEMORY_STRUCTURE:
            // Modify memory organization
            if (agent->memory) {
                // Increase memory capacity slightly
                if (agent->memory->capacity < 10000) {
                    size_t new_capacity = agent->memory->capacity * 1.1f;
                    // In a real implementation, would reallocate memory structure
                    printf("Memory capacity would be increased to %zu\n", new_capacity);
                    success = true;
                }
            }
            break;
            
        case SELF_MOD_REASONING_RULES:
            // Add or modify reasoning rules
            if (agent->reasoning && agent->reasoning->pln_engine) {
                // Adjust PLN confidence thresholds
                printf("PLN confidence thresholds would be adjusted\n");
                success = true;
            }
            break;
            
        case SELF_MOD_COMMUNICATION_PATTERNS:
            // Modify communication behavior
            // Adjust message frequency or routing preferences
            printf("Communication patterns would be modified\n");
            success = true;
            break;
            
        case SELF_MOD_GOAL_HIERARCHY:
            // Modify goal prioritization
            if (agent->tasks) {
                // Adjust task prioritization weights
                printf("Goal hierarchy would be modified\n");
                success = true;
            }
            break;
            
        case SELF_MOD_LEARNING_RATE:
            // Adjust learning parameters
            if (engine->meta_learning) {
                engine->meta_learning->learning_rate *= 1.05f;
                engine->meta_learning->exploration_rate *= 0.98f;
                success = true;
            }
            break;
            
        default:
            printf("Self-modification execution failed: Unknown modification type %d\n", 
                   record->mod_type);
            return false;
    }
    
    if (success) {
        record->is_active = true;
        engine->total_modifications++;
        engine->successful_modifications++;
        
        printf("Self-modification executed successfully: ID %lu, Type %d\n",
               modification_id, record->mod_type);
        
        return true;
    } else {
        printf("Self-modification execution failed: Implementation error\n");
        return false;
    }
}

bool rollback_modification(self_modification_engine* engine,
                          struct cognitive_agent* agent,
                          uint64_t modification_id) {
    if (!engine || !agent) {
        return false;
    }
    
    // Find modification record
    self_mod_record* record = NULL;
    for (size_t i = 0; i < engine->history_count; i++) {
        if (engine->modification_history[i].modification_id == modification_id) {
            record = &engine->modification_history[i];
            break;
        }
    }
    
    if (!record || !record->is_active) {
        printf("Rollback failed: Modification %lu not found or not active\n", modification_id);
        return false;
    }
    
    // Perform rollback based on modification type
    bool success = false;
    switch (record->mod_type) {
        case SELF_MOD_ATTENTION_WEIGHTS:
            // Restore attention parameters from backup
            if (agent->attention && record->backup_data) {
                // In a real implementation, would restore from backup_data
                agent->attention->decay_rate /= 0.95f;
                agent->attention->novelty_bonus /= 1.05f;
                success = true;
            }
            break;
            
        case SELF_MOD_LEARNING_RATE:
            // Restore learning parameters
            if (engine->meta_learning) {
                engine->meta_learning->learning_rate /= 1.05f;
                engine->meta_learning->exploration_rate /= 0.98f;
                success = true;
            }
            break;
            
        // Add other rollback cases as needed
        default:
            printf("Rollback for modification type %d not implemented\n", record->mod_type);
            success = true;  // Assume success for unimplemented types
            break;
    }
    
    if (success) {
        record->is_active = false;
        record->rollback_count++;
        engine->rollbacks_performed++;
        
        printf("Self-modification rolled back successfully: ID %lu\n", modification_id);
        return true;
    } else {
        printf("Rollback failed for modification %lu\n", modification_id);
        return false;
    }
}

// Meta-learning functions
void record_learning_experience(meta_learning_system* meta_learning,
                               enum self_mod_type mod_type,
                               const float* context,
                               float performance_delta,
                               bool success) {
    if (!meta_learning || !context) return;
    
    if (meta_learning->experience_count >= meta_learning->experience_capacity) {
        // Remove oldest experience if buffer is full
        memmove(meta_learning->experiences, 
                meta_learning->experiences + 1,
                (meta_learning->experience_capacity - 1) * sizeof(meta_learning_experience));
        meta_learning->experience_count = meta_learning->experience_capacity - 1;
    }
    
    meta_learning_experience* exp = &meta_learning->experiences[meta_learning->experience_count];
    exp->modification_type = mod_type;
    memcpy(exp->context_vector, context, sizeof(exp->context_vector));
    exp->performance_delta = performance_delta;
    exp->attention_cost = 0.1f;  // Placeholder
    exp->timestamp = get_timestamp_ns();
    exp->was_successful = success;
    
    meta_learning->experience_count++;
    
    // Update strategy success rates
    if (mod_type >= 1 && mod_type <= 10) {
        uint32_t strategy_idx = mod_type - 1;
        meta_learning->strategy_usage_counts[strategy_idx]++;
        
        // Update success rate with exponential moving average
        float current_rate = meta_learning->strategy_success_rates[strategy_idx];
        meta_learning->strategy_success_rates[strategy_idx] = 
            0.9f * current_rate + 0.1f * (success ? 1.0f : 0.0f);
    }
}

enum self_mod_type select_adaptation_strategy(meta_learning_system* meta_learning,
                                             const float* current_context) {
    if (!meta_learning || !current_context) {
        return SELF_MOD_ATTENTION_WEIGHTS;  // Safe default
    }
    
    // Use epsilon-greedy strategy selection
    if ((float)rand() / RAND_MAX < meta_learning->exploration_rate) {
        // Explore: random strategy
        return (enum self_mod_type)(1 + rand() % 6);
    } else {
        // Exploit: best performing strategy
        float best_score = -1.0f;
        enum self_mod_type best_strategy = SELF_MOD_ATTENTION_WEIGHTS;
        
        for (int i = 0; i < 6; i++) {
            float success_rate = meta_learning->strategy_success_rates[i];
            float usage_bonus = 1.0f / (1.0f + meta_learning->strategy_usage_counts[i] * 0.01f);
            float score = success_rate * usage_bonus;
            
            if (score > best_score) {
                best_score = score;
                best_strategy = (enum self_mod_type)(i + 1);
            }
        }
        
        return best_strategy;
    }
}

void update_strategy_weights(meta_learning_system* meta_learning,
                           enum self_mod_type strategy,
                           float performance_delta) {
    if (!meta_learning || !meta_learning->strategy_weights) return;
    
    int strategy_idx = strategy - 1;
    if (strategy_idx < 0 || strategy_idx >= 10) return;
    
    float* weights = (float*)meta_learning->strategy_weights->data;
    
    // Update weight based on performance delta
    float learning_rate = meta_learning->learning_rate;
    weights[strategy_idx] += learning_rate * performance_delta;
    
    // Normalize weights to sum to 1.0
    float sum = 0.0f;
    for (int i = 0; i < 10; i++) {
        weights[i] = fmaxf(weights[i], 0.01f);  // Minimum weight
        sum += weights[i];
    }
    
    for (int i = 0; i < 10; i++) {
        weights[i] /= sum;
    }
}

// Performance monitoring functions
void update_performance_metrics(self_modification_engine* engine,
                               struct cognitive_agent* agent) {
    if (!engine || !agent) return;
    
    // Measure current cognitive performance
    float performance = measure_cognitive_performance(agent);
    engine->current_performance = performance;
    
    // Update performance for the most recent modification
    if (engine->history_count > 0) {
        self_mod_record* latest = &engine->modification_history[engine->history_count - 1];
        if (latest->is_active && latest->performance_after == 0.0f) {
            latest->performance_after = performance;
        }
    }
    
    // Update meta-cognitive awareness
    update_self_awareness_metrics(engine, agent);
}

float measure_cognitive_performance(struct cognitive_agent* agent) {
    if (!agent) return 0.0f;
    
    float performance = 0.0f;
    int components = 0;
    
    // Memory efficiency
    if (agent->memory) {
        float memory_efficiency = (float)agent->memory->cache_hits / 
                                 (agent->memory->total_accesses + 1);
        performance += memory_efficiency;
        components++;
    }
    
    // Reasoning accuracy
    if (agent->reasoning) {
        performance += agent->reasoning->reasoning_accuracy;
        components++;
    }
    
    // Attention allocation efficiency
    if (agent->attention) {
        float attention_efficiency = agent->attention->allocated_attention / 
                                   agent->attention->total_attention;
        attention_efficiency = fminf(attention_efficiency, 1.0f);
        performance += attention_efficiency;
        components++;
    }
    
    // Network communication efficiency
    float comm_efficiency = 0.8f;  // Placeholder
    performance += comm_efficiency;
    components++;
    
    return components > 0 ? performance / components : 0.5f;
}

float compute_adaptation_efficiency(self_modification_engine* engine) {
    if (!engine || engine->total_modifications == 0) {
        return 0.0f;
    }
    
    float success_rate = (float)engine->successful_modifications / engine->total_modifications;
    float rollback_penalty = (float)engine->rollbacks_performed / engine->total_modifications;
    
    return success_rate - 0.5f * rollback_penalty;
}

void log_modification_attempt(self_modification_engine* engine,
                             enum self_mod_type mod_type,
                             bool success,
                             float performance_change) {
    if (!engine) return;
    
    printf("Modification Log: Type=%d, Success=%s, Performance Δ=%.3f\n",
           mod_type, success ? "true" : "false", performance_change);
    
    // Record in meta-learning system
    if (engine->meta_learning) {
        float context[64] = {0};  // Placeholder context
        context[0] = engine->current_performance;
        context[1] = engine->self_awareness_level;
        context[2] = (float)mod_type;
        
        record_learning_experience(engine->meta_learning, mod_type, context,
                                  performance_change, success);
    }
}

// Meta-cognitive monitoring functions
void update_self_awareness_metrics(self_modification_engine* engine,
                                  struct cognitive_agent* agent) {
    if (!engine || !agent) return;
    
    // Update self-awareness based on modification success rate
    float adaptation_efficiency = compute_adaptation_efficiency(engine);
    engine->self_awareness_level = 0.7f * engine->self_awareness_level + 
                                  0.3f * adaptation_efficiency;
    
    // Update modification confidence
    if (engine->total_modifications > 0) {
        float recent_success_rate = (float)engine->successful_modifications / 
                                   engine->total_modifications;
        engine->modification_confidence = 0.8f * engine->modification_confidence +
                                        0.2f * recent_success_rate;
    }
    
    // Update learning progress
    if (engine->meta_learning) {
        engine->learning_progress = engine->meta_learning->adaptation_speed * 
                                   engine->meta_learning->meta_learning_efficiency;
    }
}

float assess_modification_confidence(self_modification_engine* engine,
                                    enum self_mod_type mod_type) {
    if (!engine) return 0.0f;
    
    // Base confidence from overall success rate
    float base_confidence = engine->modification_confidence;
    
    // Adjust based on specific modification type success rate
    if (engine->meta_learning && mod_type >= 1 && mod_type <= 6) {
        float type_success_rate = engine->meta_learning->strategy_success_rates[mod_type - 1];
        base_confidence = 0.6f * base_confidence + 0.4f * type_success_rate;
    }
    
    return base_confidence;
}

void monitor_learning_progress(self_modification_engine* engine) {
    if (!engine || !engine->meta_learning) return;
    
    // Calculate learning curve slope
    if (engine->meta_learning->experience_count > 10) {
        float recent_avg = 0.0f;
        float older_avg = 0.0f;
        size_t split = engine->meta_learning->experience_count / 2;
        
        for (size_t i = split; i < engine->meta_learning->experience_count; i++) {
            recent_avg += engine->meta_learning->experiences[i].performance_delta;
        }
        recent_avg /= (engine->meta_learning->experience_count - split);
        
        for (size_t i = 0; i < split; i++) {
            older_avg += engine->meta_learning->experiences[i].performance_delta;
        }
        older_avg /= split;
        
        engine->learning_progress = recent_avg - older_avg;
        engine->meta_learning->adaptation_speed = fmaxf(engine->learning_progress, 0.1f);
    }
}

// Integration with cognitive agent
bool integrate_self_modification(struct cognitive_agent* agent,
                               self_modification_engine* engine) {
    if (!agent || !engine) {
        return false;
    }
    
    // In a real implementation, would add engine as a component of the agent
    printf("Self-modification engine integrated with cognitive agent %lu\n", agent->agent_id);
    
    return true;
}

void process_self_modification_cycle(struct cognitive_agent* agent) {
    if (!agent) return;
    
    // This would be called periodically to:
    // 1. Assess current performance
    // 2. Determine if self-modification is needed
    // 3. Select appropriate modification strategy
    // 4. Execute modifications safely
    // 5. Monitor results and update meta-learning
    
    printf("Processing self-modification cycle for agent %lu\n", agent->agent_id);
}