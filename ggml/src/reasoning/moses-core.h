#pragma once

#include "ggml.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// MOSES Program representation
struct moses_program {
    struct ggml_tensor* program_graph;    // ggml computation graph representing the program
    float* parameters;                    // Program parameters
    size_t parameter_count;               // Number of parameters
    
    // Fitness metrics
    float fitness_score;                  // Current fitness
    float complexity_penalty;            // Complexity cost
    float novelty_score;                  // Diversity measure
    
    // Genetic lineage
    uint64_t program_id;                  // Unique identifier
    uint64_t parent1_id;                  // First parent ID
    uint64_t parent2_id;                  // Second parent ID (for crossover)
    uint32_t generation;                  // Generation number
    uint32_t mutation_count;              // Number of mutations applied
    
    // Program metadata
    char* description;                    // Human-readable description
    uint64_t creation_time;               // When program was created
    uint64_t last_evaluation;             // Last fitness evaluation time
};

// MOSES Population
struct moses_population {
    struct moses_program** programs;      // Array of programs
    size_t population_size;               // Current population size
    size_t capacity;                      // Maximum capacity
    
    // Evolution parameters
    float mutation_rate;                  // Probability of mutation
    float crossover_rate;                 // Probability of crossover
    float elitism_ratio;                  // Fraction of best programs to preserve
    
    // Statistics
    float best_fitness;                   // Best fitness in population
    float average_fitness;                // Average fitness
    float diversity_measure;              // Population diversity
    uint32_t current_generation;          // Current generation number
    uint64_t total_evaluations;           // Total fitness evaluations
};

// MOSES Fitness Function
typedef float (*moses_fitness_function)(struct moses_program* program, void* context);

// MOSES Evolution Engine
struct moses_engine {
    struct ggml_context* ctx;
    struct moses_population* population;
    
    // Evolution configuration
    moses_fitness_function fitness_func;  // Fitness evaluation function
    void* fitness_context;                // Context for fitness function
    
    // Genetic operators
    float mutation_strength;              // Standard deviation for parameter mutations
    size_t tournament_size;               // Tournament selection size
    bool use_elitism;                     // Whether to use elitist selection
    
    // Performance tracking
    uint32_t generations_evolved;         // Total generations processed
    uint64_t programs_created;            // Total programs created
    float convergence_threshold;          // Fitness convergence criterion
    bool converged;                       // Whether evolution has converged
    
    // Resource management
    uint64_t max_evaluation_time_ns;     // Maximum time per fitness evaluation
    size_t max_program_complexity;        // Maximum allowed program complexity
};

// MOSES Core Functions
struct moses_engine* moses_engine_create(struct ggml_context* ctx, size_t population_size);
void moses_engine_destroy(struct moses_engine* engine);

// Population management
int moses_initialize_population(struct moses_engine* engine);
void moses_population_destroy(struct moses_population* population);

// Program operations
struct moses_program* moses_program_create(struct moses_engine* engine);
struct moses_program* moses_program_clone(struct moses_program* original);
void moses_program_destroy(struct moses_program* program);
float moses_program_evaluate_fitness(struct moses_program* program, moses_fitness_function func, void* context);

// Genetic operators
struct moses_program* moses_mutate(struct moses_program* parent, float mutation_rate);
struct moses_program* moses_crossover(struct moses_program* parent1, struct moses_program* parent2);
struct moses_program** moses_tournament_selection(struct moses_population* population, size_t tournament_size, size_t count);

// Evolution cycle
int moses_evolve_generation(struct moses_engine* engine);
int moses_run_evolution(struct moses_engine* engine, uint32_t max_generations);

// Fitness functions
float moses_default_fitness_function(struct moses_program* program, void* context);
float moses_complexity_fitness_function(struct moses_program* program, void* context);
float moses_cognitive_fitness_function(struct moses_program* program, void* context);

// Diversity and novelty
float moses_calculate_program_diversity(struct moses_population* population);
float moses_calculate_program_novelty(struct moses_program* program, struct moses_population* population);

// Configuration
void moses_set_fitness_function(struct moses_engine* engine, moses_fitness_function func, void* context);
void moses_set_mutation_rate(struct moses_engine* engine, float rate);
void moses_set_crossover_rate(struct moses_engine* engine, float rate);
void moses_set_elitism_ratio(struct moses_engine* engine, float ratio);

// Statistics and monitoring
void moses_print_population_stats(struct moses_engine* engine);
void moses_print_program_summary(struct moses_program* program);
float moses_get_convergence_rate(struct moses_engine* engine);

// Integration with cognitive agents
int moses_integrate_with_reasoning(struct moses_engine* engine, void* reasoning_engine);
struct moses_program* moses_evolve_reasoning_strategy(struct moses_engine* engine, const char* problem_description);

// Utility functions
uint64_t moses_get_timestamp(void);
float moses_random_float(void);
float moses_gaussian_random(float mean, float stddev);
void moses_seed_random(uint32_t seed);

#ifdef __cplusplus
}
#endif