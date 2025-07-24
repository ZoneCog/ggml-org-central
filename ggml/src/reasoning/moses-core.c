#include "moses-core.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

// Random number generation state
static uint32_t moses_rng_state = 12345;

// Get high-resolution timestamp
uint64_t moses_get_timestamp(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

// Seed random number generator
void moses_seed_random(uint32_t seed) {
    moses_rng_state = seed;
}

// Generate random float [0, 1)
float moses_random_float(void) {
    moses_rng_state = moses_rng_state * 1664525 + 1013904223; // Linear congruential generator
    return (float)moses_rng_state / (float)UINT32_MAX;
}

// Generate Gaussian random number using Box-Muller transform
float moses_gaussian_random(float mean, float stddev) {
    static bool has_spare = false;
    static float spare;
    
    if (has_spare) {
        has_spare = false;
        return spare * stddev + mean;
    }
    
    has_spare = true;
    float u = moses_random_float();
    float v = moses_random_float();
    float mag = stddev * sqrtf(-2.0f * logf(u));
    spare = mag * cosf(2.0f * M_PI * v);
    return mag * sinf(2.0f * M_PI * v) + mean;
}

// Create MOSES Engine
struct moses_engine* moses_engine_create(struct ggml_context* ctx, size_t population_size) {
    struct moses_engine* engine = malloc(sizeof(struct moses_engine));
    if (!engine) return NULL;
    
    engine->ctx = ctx;
    engine->population = NULL;
    engine->fitness_func = moses_default_fitness_function;
    engine->fitness_context = NULL;
    
    // Default evolution parameters
    engine->mutation_strength = 0.1f;
    engine->tournament_size = 3;
    engine->use_elitism = true;
    engine->convergence_threshold = 0.001f;
    engine->converged = false;
    
    // Performance tracking
    engine->generations_evolved = 0;
    engine->programs_created = 0;
    engine->max_evaluation_time_ns = 1000000000; // 1 second
    engine->max_program_complexity = 1000;
    
    // Initialize population
    engine->population = malloc(sizeof(struct moses_population));
    engine->population->programs = malloc(sizeof(struct moses_program*) * population_size);
    engine->population->population_size = 0;
    engine->population->capacity = population_size;
    engine->population->mutation_rate = 0.1f;
    engine->population->crossover_rate = 0.7f;
    engine->population->elitism_ratio = 0.1f;
    engine->population->best_fitness = -INFINITY;
    engine->population->average_fitness = 0.0f;
    engine->population->diversity_measure = 0.0f;
    engine->population->current_generation = 0;
    engine->population->total_evaluations = 0;
    
    // Seed random number generator
    moses_seed_random((uint32_t)time(NULL));
    
    return engine;
}

// Destroy MOSES Engine
void moses_engine_destroy(struct moses_engine* engine) {
    if (!engine) return;
    
    if (engine->population) {
        moses_population_destroy(engine->population);
    }
    
    free(engine);
}

// Destroy population
void moses_population_destroy(struct moses_population* population) {
    if (!population) return;
    
    for (size_t i = 0; i < population->population_size; i++) {
        moses_program_destroy(population->programs[i]);
    }
    
    free(population->programs);
    free(population);
}

// Create a new program
struct moses_program* moses_program_create(struct moses_engine* engine) {
    if (!engine) return NULL;
    
    struct moses_program* program = malloc(sizeof(struct moses_program));
    if (!program) return NULL;
    
    // Initialize basic structure
    program->program_graph = NULL; // Will be set based on problem type
    program->parameter_count = 10 + (size_t)(moses_random_float() * 20); // 10-30 parameters
    program->parameters = malloc(sizeof(float) * program->parameter_count);
    
    // Initialize parameters randomly
    for (size_t i = 0; i < program->parameter_count; i++) {
        program->parameters[i] = moses_gaussian_random(0.0f, 1.0f);
    }
    
    // Initialize fitness metrics
    program->fitness_score = -INFINITY;
    program->complexity_penalty = (float)program->parameter_count * 0.01f;
    program->novelty_score = 0.0f;
    
    // Set genetic lineage
    program->program_id = ++engine->programs_created;
    program->parent1_id = 0;
    program->parent2_id = 0;
    program->generation = engine->population->current_generation;
    program->mutation_count = 0;
    
    // Set metadata
    program->description = malloc(64);
    snprintf(program->description, 64, "Program_%lu_Gen_%u", program->program_id, program->generation);
    program->creation_time = moses_get_timestamp();
    program->last_evaluation = 0;
    
    return program;
}

// Clone a program
struct moses_program* moses_program_clone(struct moses_program* original) {
    if (!original) return NULL;
    
    struct moses_program* clone = malloc(sizeof(struct moses_program));
    if (!clone) return NULL;
    
    // Copy all fields
    *clone = *original;
    
    // Deep copy dynamic data
    clone->parameters = malloc(sizeof(float) * original->parameter_count);
    memcpy(clone->parameters, original->parameters, sizeof(float) * original->parameter_count);
    
    clone->description = malloc(strlen(original->description) + 1);
    strcpy(clone->description, original->description);
    
    // Reset some fields for the clone
    clone->program_id = 0; // Will be set when added to population
    clone->creation_time = moses_get_timestamp();
    
    return clone;
}

// Destroy a program
void moses_program_destroy(struct moses_program* program) {
    if (!program) return;
    
    free(program->parameters);
    free(program->description);
    free(program);
}

// Default fitness function
float moses_default_fitness_function(struct moses_program* program, void* context) {
    if (!program) return -INFINITY;
    
    // Simple fitness based on parameter optimization
    float fitness = 0.0f;
    
    // Reward parameters close to target values
    for (size_t i = 0; i < program->parameter_count; i++) {
        float target = (float)i / (float)program->parameter_count; // Target pattern
        float diff = fabsf(program->parameters[i] - target);
        fitness -= diff; // Minimize difference
    }
    
    // Apply complexity penalty
    fitness -= program->complexity_penalty;
    
    return fitness;
}

// Complexity-aware fitness function
float moses_complexity_fitness_function(struct moses_program* program, void* context) {
    if (!program) return -INFINITY;
    
    float base_fitness = moses_default_fitness_function(program, context);
    
    // Additional complexity penalty
    float complexity = (float)program->parameter_count;
    float complexity_penalty = complexity * complexity * 0.001f; // Quadratic penalty
    
    return base_fitness - complexity_penalty;
}

// Cognitive fitness function (for reasoning optimization)
float moses_cognitive_fitness_function(struct moses_program* program, void* context) {
    if (!program) return -INFINITY;
    
    // This would evaluate how well the program performs cognitive tasks
    // For now, we simulate with a more complex function
    
    float fitness = 0.0f;
    
    // Reward certain patterns in parameters that might represent cognitive structures
    for (size_t i = 1; i < program->parameter_count; i++) {
        // Reward local coherence (similar nearby parameters)
        float coherence = 1.0f - fabsf(program->parameters[i] - program->parameters[i-1]);
        fitness += coherence * 0.1f;
        
        // Reward oscillatory patterns (might represent neural oscillations)
        if (i >= 2) {
            float oscillation = cosf(program->parameters[i] * 2.0f * M_PI);
            fitness += oscillation * 0.05f;
        }
    }
    
    // Apply complexity penalty
    fitness -= program->complexity_penalty * 2.0f;
    
    return fitness;
}

// Evaluate program fitness
float moses_program_evaluate_fitness(struct moses_program* program, moses_fitness_function func, void* context) {
    if (!program || !func) return -INFINITY;
    
    uint64_t start_time = moses_get_timestamp();
    
    program->fitness_score = func(program, context);
    program->last_evaluation = moses_get_timestamp();
    
    uint64_t evaluation_time = program->last_evaluation - start_time;
    
    // Penalize programs that take too long to evaluate
    if (evaluation_time > 1000000) { // 1ms threshold
        program->fitness_score *= 0.9f;
    }
    
    return program->fitness_score;
}

// Mutate a program
struct moses_program* moses_mutate(struct moses_program* parent, float mutation_rate) {
    if (!parent) return NULL;
    
    struct moses_program* mutant = moses_program_clone(parent);
    if (!mutant) return NULL;
    
    // Apply mutations to parameters
    for (size_t i = 0; i < mutant->parameter_count; i++) {
        if (moses_random_float() < mutation_rate) {
            float mutation = moses_gaussian_random(0.0f, 0.1f);
            mutant->parameters[i] += mutation;
            
            // Clamp to reasonable range
            if (mutant->parameters[i] > 10.0f) mutant->parameters[i] = 10.0f;
            if (mutant->parameters[i] < -10.0f) mutant->parameters[i] = -10.0f;
        }
    }
    
    // Update metadata
    mutant->parent1_id = parent->program_id;
    mutant->parent2_id = 0;
    mutant->mutation_count = parent->mutation_count + 1;
    mutant->fitness_score = -INFINITY; // Need to re-evaluate
    
    // Update description
    free(mutant->description);
    mutant->description = malloc(80);
    snprintf(mutant->description, 80, "Mutant_of_%lu_Gen_%u", parent->program_id, mutant->generation);
    
    return mutant;
}

// Crossover two programs
struct moses_program* moses_crossover(struct moses_program* parent1, struct moses_program* parent2) {
    if (!parent1 || !parent2) return NULL;
    
    struct moses_program* offspring = moses_program_clone(parent1);
    if (!offspring) return NULL;
    
    size_t min_params = (parent1->parameter_count < parent2->parameter_count) ? 
                        parent1->parameter_count : parent2->parameter_count;
    
    // Uniform crossover
    for (size_t i = 0; i < min_params; i++) {
        if (moses_random_float() < 0.5f) {
            offspring->parameters[i] = parent2->parameters[i];
        }
    }
    
    // Update metadata
    offspring->parent1_id = parent1->program_id;
    offspring->parent2_id = parent2->program_id;
    offspring->mutation_count = 0;
    offspring->fitness_score = -INFINITY; // Need to re-evaluate
    
    // Update description
    free(offspring->description);
    offspring->description = malloc(100);
    snprintf(offspring->description, 100, "Crossover_%lu_x_%lu_Gen_%u", 
             parent1->program_id, parent2->program_id, offspring->generation);
    
    return offspring;
}

// Tournament selection
struct moses_program** moses_tournament_selection(struct moses_population* population, 
                                                  size_t tournament_size, size_t count) {
    if (!population || tournament_size == 0 || count == 0) return NULL;
    
    struct moses_program** selected = malloc(sizeof(struct moses_program*) * count);
    
    for (size_t i = 0; i < count; i++) {
        struct moses_program* best = NULL;
        float best_fitness = -INFINITY;
        
        // Run tournament
        for (size_t j = 0; j < tournament_size; j++) {
            size_t index = (size_t)(moses_random_float() * population->population_size);
            struct moses_program* candidate = population->programs[index];
            
            if (candidate->fitness_score > best_fitness) {
                best = candidate;
                best_fitness = candidate->fitness_score;
            }
        }
        
        selected[i] = best;
    }
    
    return selected;
}

// Initialize population with random programs
int moses_initialize_population(struct moses_engine* engine) {
    if (!engine || !engine->population) return -1;
    
    for (size_t i = 0; i < engine->population->capacity; i++) {
        struct moses_program* program = moses_program_create(engine);
        if (!program) return -1;
        
        program->program_id = ++engine->programs_created;
        
        // Evaluate initial fitness
        moses_program_evaluate_fitness(program, engine->fitness_func, engine->fitness_context);
        
        engine->population->programs[i] = program;
        engine->population->population_size++;
        engine->population->total_evaluations++;
    }
    
    return 0;
}

// Calculate population diversity
float moses_calculate_program_diversity(struct moses_population* population) {
    if (!population || population->population_size < 2) return 0.0f;
    
    float total_distance = 0.0f;
    size_t comparisons = 0;
    
    // Calculate average pairwise distance
    for (size_t i = 0; i < population->population_size; i++) {
        for (size_t j = i + 1; j < population->population_size; j++) {
            struct moses_program* prog1 = population->programs[i];
            struct moses_program* prog2 = population->programs[j];
            
            float distance = 0.0f;
            size_t min_params = (prog1->parameter_count < prog2->parameter_count) ?
                               prog1->parameter_count : prog2->parameter_count;
            
            for (size_t k = 0; k < min_params; k++) {
                float diff = prog1->parameters[k] - prog2->parameters[k];
                distance += diff * diff;
            }
            
            distance = sqrtf(distance);
            total_distance += distance;
            comparisons++;
        }
    }
    
    return comparisons > 0 ? total_distance / comparisons : 0.0f;
}

// Evolve one generation
int moses_evolve_generation(struct moses_engine* engine) {
    if (!engine || !engine->population) return -1;
    
    struct moses_population* pop = engine->population;
    
    // Calculate current population statistics
    float total_fitness = 0.0f;
    float best_fitness = -INFINITY;
    
    for (size_t i = 0; i < pop->population_size; i++) {
        float fitness = pop->programs[i]->fitness_score;
        total_fitness += fitness;
        if (fitness > best_fitness) {
            best_fitness = fitness;
        }
    }
    
    pop->average_fitness = total_fitness / pop->population_size;
    pop->best_fitness = best_fitness;
    pop->diversity_measure = moses_calculate_program_diversity(pop);
    
    // Create new generation
    struct moses_program** new_generation = malloc(sizeof(struct moses_program*) * pop->capacity);
    size_t new_count = 0;
    
    // Elitism: preserve best programs
    if (engine->use_elitism) {
        size_t elite_count = (size_t)(pop->population_size * pop->elitism_ratio);
        
        // Simple selection of best programs (could be improved with proper sorting)
        for (size_t i = 0; i < elite_count && new_count < pop->capacity; i++) {
            float best_in_pop = -INFINITY;
            size_t best_index = 0;
            
            for (size_t j = 0; j < pop->population_size; j++) {
                if (pop->programs[j]->fitness_score > best_in_pop) {
                    // Check if already selected
                    bool already_selected = false;
                    for (size_t k = 0; k < new_count; k++) {
                        if (new_generation[k] == pop->programs[j]) {
                            already_selected = true;
                            break;
                        }
                    }
                    
                    if (!already_selected) {
                        best_in_pop = pop->programs[j]->fitness_score;
                        best_index = j;
                    }
                }
            }
            
            new_generation[new_count++] = moses_program_clone(pop->programs[best_index]);
        }
    }
    
    // Generate offspring
    while (new_count < pop->capacity) {
        if (moses_random_float() < pop->crossover_rate && new_count + 1 < pop->capacity) {
            // Crossover
            struct moses_program** parents = moses_tournament_selection(pop, engine->tournament_size, 2);
            struct moses_program* offspring = moses_crossover(parents[0], parents[1]);
            
            if (offspring) {
                offspring->program_id = ++engine->programs_created;
                offspring->generation = pop->current_generation + 1;
                
                // Evaluate fitness
                moses_program_evaluate_fitness(offspring, engine->fitness_func, engine->fitness_context);
                pop->total_evaluations++;
                
                new_generation[new_count++] = offspring;
            }
            
            free(parents);
        } else {
            // Mutation
            struct moses_program** parents = moses_tournament_selection(pop, engine->tournament_size, 1);
            struct moses_program* mutant = moses_mutate(parents[0], pop->mutation_rate);
            
            if (mutant) {
                mutant->program_id = ++engine->programs_created;
                mutant->generation = pop->current_generation + 1;
                
                // Evaluate fitness
                moses_program_evaluate_fitness(mutant, engine->fitness_func, engine->fitness_context);
                pop->total_evaluations++;
                
                new_generation[new_count++] = mutant;
            }
            
            free(parents);
        }
    }
    
    // Replace old generation
    for (size_t i = 0; i < pop->population_size; i++) {
        moses_program_destroy(pop->programs[i]);
    }
    
    free(pop->programs);
    pop->programs = new_generation;
    pop->population_size = new_count;
    pop->current_generation++;
    engine->generations_evolved++;
    
    return 0;
}

// Configuration functions
void moses_set_fitness_function(struct moses_engine* engine, moses_fitness_function func, void* context) {
    if (engine) {
        engine->fitness_func = func;
        engine->fitness_context = context;
    }
}

void moses_set_mutation_rate(struct moses_engine* engine, float rate) {
    if (engine && engine->population) {
        engine->population->mutation_rate = rate;
    }
}

void moses_set_crossover_rate(struct moses_engine* engine, float rate) {
    if (engine && engine->population) {
        engine->population->crossover_rate = rate;
    }
}

void moses_set_elitism_ratio(struct moses_engine* engine, float ratio) {
    if (engine && engine->population) {
        engine->population->elitism_ratio = ratio;
    }
}

// Print statistics
void moses_print_population_stats(struct moses_engine* engine) {
    if (!engine || !engine->population) {
        printf("MOSES engine not initialized\n");
        return;
    }
    
    struct moses_population* pop = engine->population;
    
    printf("MOSES Population Statistics:\n");
    printf("  Generation: %u\n", pop->current_generation);
    printf("  Population size: %zu\n", pop->population_size);
    printf("  Best fitness: %.4f\n", pop->best_fitness);
    printf("  Average fitness: %.4f\n", pop->average_fitness);
    printf("  Diversity: %.4f\n", pop->diversity_measure);
    printf("  Total evaluations: %lu\n", pop->total_evaluations);
    printf("  Programs created: %lu\n", engine->programs_created);
}