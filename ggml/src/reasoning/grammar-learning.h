#pragma once

#include "advanced-grammar.h"
#include "ggml.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct cognitive_agent;

// Grammar learning types
enum learning_method {
    LEARNING_METHOD_INDUCTION = 1,
    LEARNING_METHOD_EVOLUTION = 2,
    LEARNING_METHOD_STATISTICAL = 3,
    LEARNING_METHOD_NEURAL = 4
};

// Grammar evolution parameters
typedef struct evolution_parameters {
    float mutation_rate;           // Probability of mutation
    float crossover_rate;          // Probability of crossover
    float selection_pressure;      // Selection strength
    uint32_t population_size;      // Size of grammar population
    uint32_t max_generations;      // Maximum evolution cycles
    float fitness_threshold;       // Convergence threshold
    uint32_t tournament_size;      // Tournament selection size
} evolution_parameters;

// Grammar population for evolution
typedef struct grammar_population {
    advanced_grammar** grammars;   // Array of grammars
    float* fitness_scores;         // Fitness for each grammar
    size_t population_size;        // Current population size
    size_t capacity;              // Maximum population capacity
    uint32_t current_generation;   // Current generation number
    evolution_parameters* params;  // Evolution parameters
} grammar_population;

// Usage pattern tracking for learning
typedef struct usage_pattern {
    char* pattern_string;          // Pattern text
    float frequency;               // How often pattern occurs
    float effectiveness;           // How well pattern works
    uint64_t first_seen;          // When first observed
    uint64_t last_seen;           // When last observed
    uint32_t usage_count;         // Total usage count
    struct usage_pattern* next;    // Linked list
} usage_pattern;

// Grammar learning engine
typedef struct grammar_learning_engine {
    advanced_grammar* base_grammar;        // Base grammar being evolved
    grammar_population* population;        // Evolution population
    usage_pattern* patterns;              // Tracked usage patterns
    size_t pattern_count;                 // Number of patterns
    enum learning_method method;          // Current learning method
    float learning_rate;                  // Learning rate parameter
    uint32_t training_cycles;             // Training iterations completed
    float current_fitness;                // Current best fitness
    bool learning_active;                 // Whether learning is active
} grammar_learning_engine;

// Example data for grammar induction
typedef struct grammar_example {
    char* input_string;            // Input example
    char* expected_output;         // Expected parse output
    float importance_weight;       // Weight for this example
    bool positive_example;         // True for positive examples
} grammar_example;

// Grammar sharing mechanism
typedef struct grammar_sharing_session {
    struct cognitive_agent** agents;      // Participating agents
    size_t agent_count;                   // Number of agents
    advanced_grammar** shared_grammars;   // Grammars being shared
    size_t grammar_count;                 // Number of shared grammars
    float sharing_threshold;              // Minimum similarity for sharing
    uint32_t sharing_rounds;              // Number of sharing rounds
} grammar_sharing_session;

// Function declarations

// Grammar learning engine management
grammar_learning_engine* create_grammar_learning_engine(advanced_grammar* base_grammar, enum learning_method method);
void destroy_grammar_learning_engine(grammar_learning_engine* engine);
bool configure_learning_parameters(grammar_learning_engine* engine, float learning_rate, uint32_t max_cycles);

// Grammar induction from examples
bool induce_grammar_from_examples(grammar_learning_engine* engine, grammar_example* examples, size_t example_count);
bool add_training_example(grammar_learning_engine* engine, const char* input, const char* expected_output, float weight);
bool extract_patterns_from_examples(grammar_learning_engine* engine, grammar_example* examples, size_t example_count);
bool generate_rules_from_patterns(grammar_learning_engine* engine, float confidence_threshold);

// Grammar evolution through usage patterns
bool track_grammar_usage(grammar_learning_engine* engine, const char* usage_context, float effectiveness);
bool analyze_usage_patterns(grammar_learning_engine* engine);
bool evolve_grammar_from_usage(grammar_learning_engine* engine);
bool update_rule_fitness(grammar_learning_engine* engine, uint32_t rule_id, float performance_change);

// Population-based grammar evolution
grammar_population* create_grammar_population(size_t population_size, evolution_parameters* params);
void destroy_grammar_population(grammar_population* population);
bool initialize_random_population(grammar_population* population, advanced_grammar* seed_grammar);
bool evaluate_population_fitness(grammar_population* population, grammar_example* test_examples, size_t test_count);
bool evolve_population_generation(grammar_population* population);
advanced_grammar* get_best_grammar(grammar_population* population);

// Grammar crossover and mutation operations
bool crossover_grammars(advanced_grammar* parent1, advanced_grammar* parent2, advanced_grammar* child);
bool mutate_grammar(advanced_grammar* grammar, float mutation_rate);
bool tournament_selection(grammar_population* population, size_t tournament_size, size_t* selected_index);

// Grammar sharing between agents
grammar_sharing_session* create_grammar_sharing_session(struct cognitive_agent** agents, size_t agent_count);
void destroy_grammar_sharing_session(grammar_sharing_session* session);
bool share_grammar_knowledge(grammar_sharing_session* session, float similarity_threshold);
bool broadcast_grammar_update(grammar_sharing_session* session, advanced_grammar* grammar, struct cognitive_agent* sender);
bool merge_shared_grammars(struct cognitive_agent* agent, advanced_grammar** grammars, size_t grammar_count, float merge_weight);

// Grammar validation and testing for learning
bool validate_learned_grammar(grammar_learning_engine* engine, grammar_example* validation_examples, size_t validation_count);
bool test_grammar_generalization(grammar_learning_engine* engine, const char** novel_inputs, size_t input_count, float* generalization_score);
bool measure_learning_progress(grammar_learning_engine* engine, float* improvement_rate);

// Statistical grammar learning
bool collect_grammar_statistics(grammar_learning_engine* engine, const char** corpus, size_t corpus_size);
bool build_ngram_model(grammar_learning_engine* engine, uint32_t n, float smoothing_factor);
bool estimate_rule_probabilities(grammar_learning_engine* engine);

// Online learning and adaptation
bool enable_online_learning(grammar_learning_engine* engine, float adaptation_rate);
bool process_feedback(grammar_learning_engine* engine, const char* input, const char* actual_output, const char* desired_output);
bool adapt_grammar_online(grammar_learning_engine* engine, float feedback_strength);

// Grammar learning metrics
bool compute_learning_metrics(grammar_learning_engine* engine, float* accuracy, float* convergence, float* diversity);
bool log_learning_progress(grammar_learning_engine* engine, const char* log_file);
bool save_learning_checkpoint(grammar_learning_engine* engine, const char* checkpoint_file);
bool load_learning_checkpoint(grammar_learning_engine* engine, const char* checkpoint_file);

// Advanced learning techniques
bool implement_grammar_distillation(grammar_learning_engine* engine, advanced_grammar* teacher_grammar, float distillation_temperature);
bool apply_regularization(grammar_learning_engine* engine, float regularization_strength);
bool use_meta_learning(grammar_learning_engine* engine, grammar_learning_engine** meta_engines, size_t meta_count);

#ifdef __cplusplus
}
#endif