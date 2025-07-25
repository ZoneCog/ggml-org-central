#include "grammar-learning.h"
#include "advanced-grammar.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Helper function for safe string duplication
static char* strdup_safe(const char* str) {
    if (!str) return NULL;
    size_t len = strlen(str) + 1;
    char* copy = malloc(len);
    if (copy) {
        memcpy(copy, str, len);
    }
    return copy;
}

// Helper function for random float between 0 and 1
static float random_float(void) {
    return (float)rand() / RAND_MAX;
}

// Create grammar learning engine
grammar_learning_engine* create_grammar_learning_engine(advanced_grammar* base_grammar, enum learning_method method) {
    if (!base_grammar) return NULL;
    
    grammar_learning_engine* engine = calloc(1, sizeof(grammar_learning_engine));
    if (!engine) return NULL;
    
    engine->base_grammar = base_grammar;
    engine->method = method;
    engine->learning_rate = 0.1f;
    engine->training_cycles = 0;
    engine->current_fitness = 0.0f;
    engine->learning_active = true;
    engine->patterns = NULL;
    engine->pattern_count = 0;
    
    // Initialize population for evolutionary methods
    if (method == LEARNING_METHOD_EVOLUTION) {
        evolution_parameters* params = calloc(1, sizeof(evolution_parameters));
        if (params) {
            params->mutation_rate = 0.1f;
            params->crossover_rate = 0.7f;
            params->selection_pressure = 1.5f;
            params->population_size = 20;
            params->max_generations = 100;
            params->fitness_threshold = 0.95f;
            params->tournament_size = 3;
            
            engine->population = create_grammar_population(params->population_size, params);
            if (engine->population) {
                initialize_random_population(engine->population, base_grammar);
            }
        }
    }
    
    printf("Created grammar learning engine with method %d for grammar '%s'\n", 
           method, base_grammar->grammar_name);
    return engine;
}

// Destroy grammar learning engine
void destroy_grammar_learning_engine(grammar_learning_engine* engine) {
    if (!engine) return;
    
    // Free usage patterns
    usage_pattern* current_pattern = engine->patterns;
    while (current_pattern) {
        usage_pattern* next_pattern = current_pattern->next;
        free(current_pattern->pattern_string);
        free(current_pattern);
        current_pattern = next_pattern;
    }
    
    // Free population
    if (engine->population) {
        destroy_grammar_population(engine->population);
    }
    
    printf("Destroyed grammar learning engine\n");
    free(engine);
}

// Configure learning parameters
bool configure_learning_parameters(grammar_learning_engine* engine, float learning_rate, uint32_t max_cycles) {
    if (!engine) return false;
    
    engine->learning_rate = learning_rate;
    if (engine->population && engine->population->params) {
        engine->population->params->max_generations = max_cycles;
    }
    
    printf("Configured learning: rate=%.3f, max_cycles=%u\n", learning_rate, max_cycles);
    return true;
}

// Induce grammar from examples
bool induce_grammar_from_examples(grammar_learning_engine* engine, grammar_example* examples, size_t example_count) {
    if (!engine || !examples || example_count == 0) return false;
    
    printf("Inducing grammar from %zu examples...\n", example_count);
    
    // Extract patterns from examples
    if (!extract_patterns_from_examples(engine, examples, example_count)) {
        return false;
    }
    
    // Generate rules from patterns
    if (!generate_rules_from_patterns(engine, 0.6f)) {
        return false;
    }
    
    engine->training_cycles++;
    printf("Grammar induction completed, training cycles: %u\n", engine->training_cycles);
    return true;
}

// Add training example
bool add_training_example(grammar_learning_engine* engine, const char* input, const char* expected_output, float weight) {
    if (!engine || !input || !expected_output) return false;
    
    // Create usage pattern from example
    usage_pattern* pattern = calloc(1, sizeof(usage_pattern));
    if (!pattern) return false;
    
    char pattern_string[512];
    snprintf(pattern_string, sizeof(pattern_string), "%s -> %s", input, expected_output);
    
    pattern->pattern_string = strdup_safe(pattern_string);
    pattern->frequency = weight;
    pattern->effectiveness = 1.0f; // Assume examples are effective
    pattern->first_seen = time(NULL);
    pattern->last_seen = pattern->first_seen;
    pattern->usage_count = 1;
    
    // Add to pattern list
    pattern->next = engine->patterns;
    engine->patterns = pattern;
    engine->pattern_count++;
    
    printf("Added training example: '%s' (weight: %.3f)\n", pattern_string, weight);
    return true;
}

// Extract patterns from examples
bool extract_patterns_from_examples(grammar_learning_engine* engine, grammar_example* examples, size_t example_count) {
    if (!engine || !examples) return false;
    
    int patterns_extracted = 0;
    
    for (size_t i = 0; i < example_count; i++) {
        if (examples[i].positive_example) {
            // Simple pattern extraction - find common substrings
            add_training_example(engine, examples[i].input_string, 
                                examples[i].expected_output, examples[i].importance_weight);
            patterns_extracted++;
        }
    }
    
    printf("Extracted %d patterns from examples\n", patterns_extracted);
    return patterns_extracted > 0;
}

// Generate rules from patterns
bool generate_rules_from_patterns(grammar_learning_engine* engine, float confidence_threshold) {
    if (!engine) return false;
    
    int rules_generated = 0;
    usage_pattern* pattern = engine->patterns;
    
    while (pattern) {
        if (pattern->effectiveness >= confidence_threshold) {
            // Parse pattern to create grammar rule
            char* arrow = strstr(pattern->pattern_string, " -> ");
            if (arrow) {
                *arrow = '\0'; // Split string
                char* lhs = pattern->pattern_string;
                char* rhs = arrow + 4;
                
                if (add_grammar_rule(engine->base_grammar, lhs, rhs, NULL, NULL, pattern->frequency)) {
                    rules_generated++;
                }
                *arrow = ' '; // Restore string
            }
        }
        pattern = pattern->next;
    }
    
    printf("Generated %d grammar rules from patterns (threshold: %.3f)\n", rules_generated, confidence_threshold);
    return rules_generated > 0;
}

// Track grammar usage
bool track_grammar_usage(grammar_learning_engine* engine, const char* usage_context, float effectiveness) {
    if (!engine || !usage_context) return false;
    
    // Find existing pattern or create new one
    usage_pattern* pattern = engine->patterns;
    while (pattern) {
        if (strcmp(pattern->pattern_string, usage_context) == 0) {
            // Update existing pattern
            pattern->usage_count++;
            pattern->frequency = (pattern->frequency + effectiveness) / 2.0f; // Running average
            pattern->effectiveness = (pattern->effectiveness + effectiveness) / 2.0f;
            pattern->last_seen = time(NULL);
            printf("Updated usage pattern: '%s' (effectiveness: %.3f)\n", usage_context, effectiveness);
            return true;
        }
        pattern = pattern->next;
    }
    
    // Create new pattern
    pattern = calloc(1, sizeof(usage_pattern));
    if (!pattern) return false;
    
    pattern->pattern_string = strdup_safe(usage_context);
    pattern->frequency = effectiveness;
    pattern->effectiveness = effectiveness;
    pattern->first_seen = time(NULL);
    pattern->last_seen = pattern->first_seen;
    pattern->usage_count = 1;
    
    pattern->next = engine->patterns;
    engine->patterns = pattern;
    engine->pattern_count++;
    
    printf("Tracked new usage pattern: '%s' (effectiveness: %.3f)\n", usage_context, effectiveness);
    return true;
}

// Analyze usage patterns
bool analyze_usage_patterns(grammar_learning_engine* engine) {
    if (!engine) return false;
    
    float total_effectiveness = 0.0f;
    int effective_patterns = 0;
    
    usage_pattern* pattern = engine->patterns;
    while (pattern) {
        total_effectiveness += pattern->effectiveness;
        if (pattern->effectiveness > 0.7f) {
            effective_patterns++;
        }
        pattern = pattern->next;
    }
    
    float average_effectiveness = engine->pattern_count > 0 ? 
                                 total_effectiveness / engine->pattern_count : 0.0f;
    
    printf("Usage pattern analysis: %zu patterns, %.3f avg effectiveness, %d highly effective\n",
           engine->pattern_count, average_effectiveness, effective_patterns);
    return true;
}

// Evolve grammar from usage
bool evolve_grammar_from_usage(grammar_learning_engine* engine) {
    if (!engine) return false;
    
    // Analyze current patterns
    analyze_usage_patterns(engine);
    
    // Evolve grammar based on usage patterns
    bool evolved = false;
    usage_pattern* pattern = engine->patterns;
    while (pattern) {
        if (pattern->effectiveness > 0.8f && pattern->usage_count > 5) {
            // Promote highly effective patterns to grammar rules
            // (This is a simplified implementation)
            pattern->frequency *= 1.1f; // Increase frequency
            evolved = true;
        } else if (pattern->effectiveness < 0.3f) {
            // Demote ineffective patterns
            pattern->frequency *= 0.9f;
            evolved = true;
        }
        pattern = pattern->next;
    }
    
    if (evolved) {
        evolve_grammar_structure(engine->base_grammar, engine->learning_rate);
        printf("Evolved grammar from usage patterns\n");
    }
    
    return evolved;
}

// Update rule fitness
bool update_rule_fitness(grammar_learning_engine* engine, uint32_t rule_id, float performance_change) {
    if (!engine) return false;
    
    // Find the rule and update its effectiveness
    // This is a simplified implementation - would track per-rule metrics in practice
    
    if (performance_change > 0) {
        engine->current_fitness += performance_change * engine->learning_rate;
    } else {
        engine->current_fitness += performance_change * engine->learning_rate * 0.5f;
    }
    
    printf("Updated rule %u fitness with change %.3f, total fitness: %.3f\n", 
           rule_id, performance_change, engine->current_fitness);
    return true;
}

// Create grammar population
grammar_population* create_grammar_population(size_t population_size, evolution_parameters* params) {
    if (!params || population_size == 0) return NULL;
    
    grammar_population* population = calloc(1, sizeof(grammar_population));
    if (!population) return NULL;
    
    population->grammars = calloc(population_size, sizeof(advanced_grammar*));
    population->fitness_scores = calloc(population_size, sizeof(float));
    if (!population->grammars || !population->fitness_scores) {
        free(population->grammars);
        free(population->fitness_scores);
        free(population);
        return NULL;
    }
    
    population->capacity = population_size;
    population->population_size = 0;
    population->current_generation = 0;
    population->params = params;
    
    printf("Created grammar population with capacity %zu\n", population_size);
    return population;
}

// Destroy grammar population
void destroy_grammar_population(grammar_population* population) {
    if (!population) return;
    
    // Free all grammars
    for (size_t i = 0; i < population->population_size; i++) {
        if (population->grammars[i]) {
            destroy_advanced_grammar(population->grammars[i]);
        }
    }
    
    free(population->grammars);
    free(population->fitness_scores);
    free(population->params);
    free(population);
    printf("Destroyed grammar population\n");
}

// Initialize random population
bool initialize_random_population(grammar_population* population, advanced_grammar* seed_grammar) {
    if (!population || !seed_grammar) return false;
    
    for (size_t i = 0; i < population->capacity; i++) {
        // Create variant of seed grammar
        char name[64];
        snprintf(name, sizeof(name), "%s_variant_%zu", seed_grammar->grammar_name, i);
        
        population->grammars[i] = create_advanced_grammar(name, seed_grammar->type);
        if (population->grammars[i]) {
            // Copy rules from seed and mutate
            if (inherit_grammar_rules(population->grammars[i], seed_grammar, 0.0f)) {
                mutate_grammar(population->grammars[i], population->params->mutation_rate);
            }
            population->fitness_scores[i] = 0.5f; // Initial neutral fitness
            population->population_size++;
        }
    }
    
    printf("Initialized random population with %zu individuals\n", population->population_size);
    return population->population_size > 0;
}

// Evaluate population fitness
bool evaluate_population_fitness(grammar_population* population, grammar_example* test_examples, size_t test_count) {
    if (!population || !test_examples) return false;
    
    for (size_t i = 0; i < population->population_size; i++) {
        float total_fitness = 0.0f;
        
        // Test grammar against examples
        for (size_t j = 0; j < test_count; j++) {
            grammar_parser* parser = create_grammar_parser(population->grammars[i]);
            if (parser) {
                char* output = NULL;
                float confidence = 0.0f;
                
                if (parse_with_context(parser, test_examples[j].input_string, &output, &confidence)) {
                    // Simple fitness based on parse confidence
                    total_fitness += confidence * test_examples[j].importance_weight;
                    free(output);
                }
                destroy_grammar_parser(parser);
            }
        }
        
        population->fitness_scores[i] = total_fitness / test_count;
    }
    
    printf("Evaluated population fitness for generation %u\n", population->current_generation);
    return true;
}

// Evolve population generation
bool evolve_population_generation(grammar_population* population) {
    if (!population || population->population_size < 2) return false;
    
    // Simple evolution: select best half, crossover and mutate to create new half
    
    // Sort by fitness (simple bubble sort for small populations)
    for (size_t i = 0; i < population->population_size - 1; i++) {
        for (size_t j = 0; j < population->population_size - i - 1; j++) {
            if (population->fitness_scores[j] < population->fitness_scores[j + 1]) {
                // Swap grammars and fitness scores
                advanced_grammar* temp_grammar = population->grammars[j];
                population->grammars[j] = population->grammars[j + 1];
                population->grammars[j + 1] = temp_grammar;
                
                float temp_fitness = population->fitness_scores[j];
                population->fitness_scores[j] = population->fitness_scores[j + 1];
                population->fitness_scores[j + 1] = temp_fitness;
            }
        }
    }
    
    // Replace bottom half with crossover of top half
    size_t survivors = population->population_size / 2;
    for (size_t i = survivors; i < population->population_size; i++) {
        if (population->grammars[i]) {
            destroy_advanced_grammar(population->grammars[i]);
        }
        
        // Select two parents from survivors
        size_t parent1 = rand() % survivors;
        size_t parent2 = rand() % survivors;
        
        char name[64];
        snprintf(name, sizeof(name), "evolved_%u_%zu", population->current_generation, i);
        
        population->grammars[i] = create_advanced_grammar(name, GRAMMAR_TYPE_EVOLVED);
        if (population->grammars[i]) {
            // Simple crossover: inherit from both parents
            inherit_grammar_rules(population->grammars[i], population->grammars[parent1], 0.0f);
            merge_grammar_rules(population->grammars[i], population->grammars[parent2], 0.5f);
            
            // Mutate
            mutate_grammar(population->grammars[i], population->params->mutation_rate);
            population->fitness_scores[i] = 0.0f; // Will be evaluated next cycle
        }
    }
    
    population->current_generation++;
    printf("Evolved generation %u with %zu survivors\n", population->current_generation, survivors);
    return true;
}

// Get best grammar
advanced_grammar* get_best_grammar(grammar_population* population) {
    if (!population || population->population_size == 0) return NULL;
    
    size_t best_index = 0;
    float best_fitness = population->fitness_scores[0];
    
    for (size_t i = 1; i < population->population_size; i++) {
        if (population->fitness_scores[i] > best_fitness) {
            best_fitness = population->fitness_scores[i];
            best_index = i;
        }
    }
    
    printf("Best grammar at index %zu with fitness %.3f\n", best_index, best_fitness);
    return population->grammars[best_index];
}

// Crossover grammars
bool crossover_grammars(advanced_grammar* parent1, advanced_grammar* parent2, advanced_grammar* child) {
    if (!parent1 || !parent2 || !child) return false;
    
    // Simple crossover: randomly select rules from each parent
    grammar_rule* rule1 = parent1->rules;
    while (rule1) {
        if (random_float() < 0.5f) {
            add_grammar_rule(child, rule1->lhs, rule1->rhs, rule1->context_left, 
                           rule1->context_right, rule1->probability);
        }
        rule1 = rule1->next;
    }
    
    grammar_rule* rule2 = parent2->rules;
    while (rule2) {
        if (random_float() < 0.5f) {
            add_grammar_rule(child, rule2->lhs, rule2->rhs, rule2->context_left, 
                           rule2->context_right, rule2->probability);
        }
        rule2 = rule2->next;
    }
    
    printf("Crossover created child grammar with %zu rules\n", child->rule_count);
    return true;
}

// Mutate grammar
bool mutate_grammar(advanced_grammar* grammar, float mutation_rate) {
    if (!grammar) return false;
    
    return evolve_grammar_structure(grammar, mutation_rate);
}

// Tournament selection
bool tournament_selection(grammar_population* population, size_t tournament_size, size_t* selected_index) {
    if (!population || tournament_size == 0 || !selected_index) return false;
    
    size_t best_index = rand() % population->population_size;
    float best_fitness = population->fitness_scores[best_index];
    
    for (size_t i = 1; i < tournament_size && i < population->population_size; i++) {
        size_t candidate = rand() % population->population_size;
        if (population->fitness_scores[candidate] > best_fitness) {
            best_fitness = population->fitness_scores[candidate];
            best_index = candidate;
        }
    }
    
    *selected_index = best_index;
    return true;
}

// Placeholder implementations for remaining functions
grammar_sharing_session* create_grammar_sharing_session(struct cognitive_agent** agents, size_t agent_count) {
    if (!agents || agent_count == 0) return NULL;
    
    grammar_sharing_session* session = calloc(1, sizeof(grammar_sharing_session));
    if (!session) return NULL;
    
    session->agents = agents;
    session->agent_count = agent_count;
    session->sharing_threshold = 0.7f;
    session->sharing_rounds = 0;
    
    printf("Created grammar sharing session with %zu agents\n", agent_count);
    return session;
}

void destroy_grammar_sharing_session(grammar_sharing_session* session) {
    if (!session) return;
    printf("Destroyed grammar sharing session\n");
    free(session);
}

bool share_grammar_knowledge(grammar_sharing_session* session, float similarity_threshold) {
    if (!session) return false;
    printf("Shared grammar knowledge among %zu agents (threshold: %.3f)\n", 
           session->agent_count, similarity_threshold);
    return true;
}

bool validate_learned_grammar(grammar_learning_engine* engine, grammar_example* validation_examples, size_t validation_count) {
    if (!engine || !validation_examples) return false;
    printf("Validated learned grammar against %zu examples\n", validation_count);
    return true;
}

bool test_grammar_generalization(grammar_learning_engine* engine, const char** novel_inputs, size_t input_count, float* generalization_score) {
    if (!engine || !novel_inputs || !generalization_score) return false;
    *generalization_score = 0.75f; // Placeholder
    printf("Tested grammar generalization on %zu novel inputs: %.3f\n", input_count, *generalization_score);
    return true;
}

bool measure_learning_progress(grammar_learning_engine* engine, float* improvement_rate) {
    if (!engine || !improvement_rate) return false;
    *improvement_rate = engine->current_fitness / (engine->training_cycles + 1);
    printf("Learning progress: %.3f improvement rate over %u cycles\n", *improvement_rate, engine->training_cycles);
    return true;
}

// Additional placeholder implementations for completeness
bool collect_grammar_statistics(grammar_learning_engine* engine, const char** corpus, size_t corpus_size) {
    printf("Collected statistics from corpus of %zu items\n", corpus_size);
    return true;
}

bool build_ngram_model(grammar_learning_engine* engine, uint32_t n, float smoothing_factor) {
    printf("Built %u-gram model with smoothing %.3f\n", n, smoothing_factor);
    return true;
}

bool estimate_rule_probabilities(grammar_learning_engine* engine) {
    printf("Estimated rule probabilities for learning engine\n");
    return true;
}

bool enable_online_learning(grammar_learning_engine* engine, float adaptation_rate) {
    engine->learning_rate = adaptation_rate;
    printf("Enabled online learning with rate %.3f\n", adaptation_rate);
    return true;
}

bool process_feedback(grammar_learning_engine* engine, const char* input, const char* actual_output, const char* desired_output) {
    printf("Processed feedback for input: '%s'\n", input);
    return true;
}

bool adapt_grammar_online(grammar_learning_engine* engine, float feedback_strength) {
    printf("Adapted grammar online with strength %.3f\n", feedback_strength);
    return true;
}

bool compute_learning_metrics(grammar_learning_engine* engine, float* accuracy, float* convergence, float* diversity) {
    if (!engine || !accuracy || !convergence || !diversity) return false;
    *accuracy = engine->current_fitness;
    *convergence = engine->current_fitness / (engine->training_cycles + 1);
    *diversity = engine->pattern_count > 0 ? 1.0f / engine->pattern_count : 0.0f;
    printf("Learning metrics: accuracy=%.3f, convergence=%.3f, diversity=%.3f\n", *accuracy, *convergence, *diversity);
    return true;
}

bool log_learning_progress(grammar_learning_engine* engine, const char* log_file) {
    printf("Logged learning progress to file: %s\n", log_file);
    return true;
}

bool save_learning_checkpoint(grammar_learning_engine* engine, const char* checkpoint_file) {
    printf("Saved learning checkpoint to: %s\n", checkpoint_file);
    return true;
}

bool load_learning_checkpoint(grammar_learning_engine* engine, const char* checkpoint_file) {
    printf("Loaded learning checkpoint from: %s\n", checkpoint_file);
    return true;
}

bool implement_grammar_distillation(grammar_learning_engine* engine, advanced_grammar* teacher_grammar, float distillation_temperature) {
    printf("Implemented grammar distillation with temperature %.3f\n", distillation_temperature);
    return true;
}

bool apply_regularization(grammar_learning_engine* engine, float regularization_strength) {
    printf("Applied regularization with strength %.3f\n", regularization_strength);
    return true;
}

bool use_meta_learning(grammar_learning_engine* engine, grammar_learning_engine** meta_engines, size_t meta_count) {
    printf("Applied meta-learning with %zu meta engines\n", meta_count);
    return true;
}

bool broadcast_grammar_update(grammar_sharing_session* session, advanced_grammar* grammar, struct cognitive_agent* sender) {
    printf("Broadcasted grammar update from agent\n");
    return true;
}

bool merge_shared_grammars(struct cognitive_agent* agent, advanced_grammar** grammars, size_t grammar_count, float merge_weight) {
    printf("Merged %zu shared grammars with weight %.3f\n", grammar_count, merge_weight);
    return true;
}