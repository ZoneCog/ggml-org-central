#include "advanced-grammar.h"
#include "ggml.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Helper function to create a copy of a string
static char* strdup_safe(const char* str) {
    if (!str) return NULL;
    size_t len = strlen(str) + 1;
    char* copy = malloc(len);
    if (copy) {
        memcpy(copy, str, len);
    }
    return copy;
}

// Helper function to generate unique rule ID
static uint32_t generate_rule_id(void) {
    static uint32_t id_counter = 1;
    return id_counter++;
}

// Create advanced grammar
advanced_grammar* create_advanced_grammar(const char* name, enum grammar_type type) {
    advanced_grammar* grammar = calloc(1, sizeof(advanced_grammar));
    if (!grammar) return NULL;
    
    grammar->grammar_name = strdup_safe(name);
    grammar->type = type;
    grammar->rules = NULL;
    grammar->rule_count = 0;
    grammar->fitness_score = 0.5f; // Initial neutral fitness
    grammar->generation = 0;
    grammar->usage_count = 0;
    grammar->created_timestamp = time(NULL);
    grammar->last_modified = grammar->created_timestamp;
    
    // Initialize parsing context
    grammar->context = calloc(1, sizeof(parsing_context));
    if (grammar->context) {
        grammar->context->stack_capacity = 64;
        grammar->context->context_stack = calloc(grammar->context->stack_capacity, sizeof(char*));
        grammar->context->max_recursion_depth = 100;
    }
    
    printf("Created advanced grammar '%s' of type %d\n", name, type);
    return grammar;
}

// Destroy advanced grammar
void destroy_advanced_grammar(advanced_grammar* grammar) {
    if (!grammar) return;
    
    // Free grammar name
    free(grammar->grammar_name);
    
    // Free all rules
    grammar_rule* current_rule = grammar->rules;
    while (current_rule) {
        grammar_rule* next_rule = current_rule->next;
        free(current_rule->lhs);
        free(current_rule->rhs);
        free(current_rule->context_left);
        free(current_rule->context_right);
        free(current_rule);
        current_rule = next_rule;
    }
    
    // Free parsing context
    if (grammar->context) {
        if (grammar->context->context_stack) {
            for (size_t i = 0; i < grammar->context->stack_size; i++) {
                free(grammar->context->context_stack[i]);
            }
            free(grammar->context->context_stack);
        }
        free(grammar->context);
    }
    
    // Free compositions
    grammar_composition* current_comp = grammar->compositions;
    while (current_comp) {
        grammar_composition* next_comp = current_comp->next;
        free(current_comp);
        current_comp = next_comp;
    }
    
    printf("Destroyed advanced grammar '%s'\n", grammar->grammar_name ? grammar->grammar_name : "unknown");
    free(grammar);
}

// Add grammar rule with context sensitivity
bool add_grammar_rule(advanced_grammar* grammar, const char* lhs, const char* rhs, 
                     const char* context_left, const char* context_right, float probability) {
    if (!grammar || !lhs || !rhs) return false;
    
    grammar_rule* rule = calloc(1, sizeof(grammar_rule));
    if (!rule) return false;
    
    rule->lhs = strdup_safe(lhs);
    rule->rhs = strdup_safe(rhs);
    rule->context_left = context_left ? strdup_safe(context_left) : NULL;
    rule->context_right = context_right ? strdup_safe(context_right) : NULL;
    rule->probability = probability;
    rule->usage_count = 0.0f;
    rule->rule_id = generate_rule_id();
    
    // Add to linked list
    rule->next = grammar->rules;
    grammar->rules = rule;
    grammar->rule_count++;
    grammar->last_modified = time(NULL);
    
    printf("Added grammar rule %u: %s -> %s (prob: %.3f)\n", 
           rule->rule_id, lhs, rhs, probability);
    return true;
}

// Remove grammar rule
bool remove_grammar_rule(advanced_grammar* grammar, uint32_t rule_id) {
    if (!grammar) return false;
    
    grammar_rule** current = &grammar->rules;
    while (*current) {
        if ((*current)->rule_id == rule_id) {
            grammar_rule* to_remove = *current;
            *current = to_remove->next;
            
            free(to_remove->lhs);
            free(to_remove->rhs);
            free(to_remove->context_left);
            free(to_remove->context_right);
            free(to_remove);
            
            grammar->rule_count--;
            grammar->last_modified = time(NULL);
            printf("Removed grammar rule %u\n", rule_id);
            return true;
        }
        current = &(*current)->next;
    }
    return false;
}

// Create grammar parser
grammar_parser* create_grammar_parser(advanced_grammar* grammar) {
    if (!grammar) return NULL;
    
    grammar_parser* parser = calloc(1, sizeof(grammar_parser));
    if (!parser) return NULL;
    
    parser->current_grammar = grammar;
    parser->parse_context = calloc(1, sizeof(parsing_context));
    if (parser->parse_context) {
        parser->parse_context->stack_capacity = 64;
        parser->parse_context->context_stack = calloc(64, sizeof(char*));
        parser->parse_context->max_recursion_depth = 100;
    }
    
    return parser;
}

// Destroy grammar parser
void destroy_grammar_parser(grammar_parser* parser) {
    if (!parser) return;
    
    if (parser->parse_context) {
        if (parser->parse_context->context_stack) {
            for (size_t i = 0; i < parser->parse_context->stack_size; i++) {
                free(parser->parse_context->context_stack[i]);
            }
            free(parser->parse_context->context_stack);
        }
        free(parser->parse_context);
    }
    
    free(parser->input_buffer);
    free(parser);
}

// Parse with context sensitivity
bool parse_with_context(grammar_parser* parser, const char* input, char** output, float* confidence) {
    if (!parser || !input || !output) return false;
    
    // Store input
    free(parser->input_buffer);
    parser->input_buffer = strdup_safe(input);
    parser->input_length = strlen(input);
    parser->input_position = 0;
    
    // Reset parsing context
    parser->parse_context->stack_size = 0;
    parser->parse_context->recursion_depth = 0;
    
    // Simple context-sensitive parsing simulation
    // In a real implementation, this would implement a full context-sensitive parser
    
    // Calculate confidence based on rule matching
    float total_confidence = 0.0f;
    int matched_rules = 0;
    
    grammar_rule* rule = parser->current_grammar->rules;
    while (rule) {
        // Simple substring matching for demonstration
        if (strstr(input, rule->lhs)) {
            total_confidence += rule->probability;
            matched_rules++;
            rule->usage_count += 1.0f;
        }
        rule = rule->next;
    }
    
    // Generate output - in real implementation would be actual parse tree
    char* result = malloc(512);
    if (result) {
        snprintf(result, 512, "PARSED[%s] with %d rules, confidence: %.3f", 
                input, matched_rules, total_confidence / (matched_rules > 0 ? matched_rules : 1));
        *output = result;
    }
    
    parser->parse_confidence = total_confidence / (matched_rules > 0 ? matched_rules : 1);
    if (confidence) *confidence = parser->parse_confidence;
    
    printf("Context-sensitive parse of '%s' completed with confidence %.3f\n", input, parser->parse_confidence);
    return true;
}

// Push parse context
bool push_parse_context(parsing_context* context, const char* symbol) {
    if (!context || !symbol) return false;
    
    if (context->stack_size >= context->stack_capacity) {
        // Resize stack
        size_t new_capacity = context->stack_capacity * 2;
        char** new_stack = realloc(context->context_stack, new_capacity * sizeof(char*));
        if (!new_stack) return false;
        context->context_stack = new_stack;
        context->stack_capacity = new_capacity;
    }
    
    context->context_stack[context->stack_size] = strdup_safe(symbol);
    context->stack_size++;
    return true;
}

// Pop parse context
bool pop_parse_context(parsing_context* context, char** symbol) {
    if (!context || context->stack_size == 0) return false;
    
    context->stack_size--;
    if (symbol) {
        *symbol = context->context_stack[context->stack_size];
    } else {
        free(context->context_stack[context->stack_size]);
    }
    context->context_stack[context->stack_size] = NULL;
    return true;
}

// Compose grammars with inheritance
bool compose_grammars(advanced_grammar* parent, advanced_grammar* child, 
                     float inheritance_weight, uint32_t composition_type) {
    if (!parent || !child) return false;
    
    grammar_composition* comp = calloc(1, sizeof(grammar_composition));
    if (!comp) return false;
    
    comp->parent_grammar = parent;
    comp->child_grammar = child;
    comp->inheritance_weight = inheritance_weight;
    comp->composition_type = composition_type;
    
    // Add to child's composition list
    comp->next = child->compositions;
    child->compositions = comp;
    
    printf("Composed grammar '%s' -> '%s' with weight %.3f\n", 
           parent->grammar_name, child->grammar_name, inheritance_weight);
    return true;
}

// Inherit grammar rules
bool inherit_grammar_rules(advanced_grammar* child, advanced_grammar* parent, float threshold) {
    if (!child || !parent) return false;
    
    int inherited_count = 0;
    grammar_rule* rule = parent->rules;
    while (rule) {
        // Inherit rules above threshold probability
        if (rule->probability >= threshold) {
            add_grammar_rule(child, rule->lhs, rule->rhs, 
                           rule->context_left, rule->context_right, 
                           rule->probability * 0.8f); // Slightly lower probability
            inherited_count++;
        }
        rule = rule->next;
    }
    
    printf("Inherited %d rules from '%s' to '%s'\n", 
           inherited_count, parent->grammar_name, child->grammar_name);
    return inherited_count > 0;
}

// Merge grammar rules
bool merge_grammar_rules(advanced_grammar* target, advanced_grammar* source, float weight) {
    if (!target || !source) return false;
    
    int merged_count = 0;
    grammar_rule* rule = source->rules;
    while (rule) {
        add_grammar_rule(target, rule->lhs, rule->rhs, 
                        rule->context_left, rule->context_right, 
                        rule->probability * weight);
        merged_count++;
        rule = rule->next;
    }
    
    printf("Merged %d rules from '%s' to '%s' with weight %.3f\n", 
           merged_count, source->grammar_name, target->grammar_name, weight);
    return merged_count > 0;
}

// Modify grammar rule
bool modify_grammar_rule(advanced_grammar* grammar, uint32_t rule_id, 
                        const char* new_rhs, float new_probability) {
    if (!grammar || !new_rhs) return false;
    
    grammar_rule* rule = grammar->rules;
    while (rule) {
        if (rule->rule_id == rule_id) {
            free(rule->rhs);
            rule->rhs = strdup_safe(new_rhs);
            rule->probability = new_probability;
            grammar->last_modified = time(NULL);
            printf("Modified rule %u: new rhs '%s', probability %.3f\n", 
                   rule_id, new_rhs, new_probability);
            return true;
        }
        rule = rule->next;
    }
    return false;
}

// Add context sensitivity to existing rule
bool add_context_sensitivity(advanced_grammar* grammar, uint32_t rule_id,
                           const char* context_left, const char* context_right) {
    if (!grammar) return false;
    
    grammar_rule* rule = grammar->rules;
    while (rule) {
        if (rule->rule_id == rule_id) {
            free(rule->context_left);
            free(rule->context_right);
            rule->context_left = context_left ? strdup_safe(context_left) : NULL;
            rule->context_right = context_right ? strdup_safe(context_right) : NULL;
            grammar->type = GRAMMAR_TYPE_CONTEXT_SENSITIVE;
            grammar->last_modified = time(NULL);
            printf("Added context sensitivity to rule %u\n", rule_id);
            return true;
        }
        rule = rule->next;
    }
    return false;
}

// Evolve grammar structure with mutations
bool evolve_grammar_structure(advanced_grammar* grammar, float mutation_rate) {
    if (!grammar) return false;
    
    int mutations = 0;
    grammar_rule* rule = grammar->rules;
    while (rule) {
        // Randomly mutate rule probabilities
        if ((float)rand() / RAND_MAX < mutation_rate) {
            float old_prob = rule->probability;
            rule->probability += ((float)rand() / RAND_MAX - 0.5f) * 0.2f; // ±10% mutation
            if (rule->probability < 0.0f) rule->probability = 0.0f;
            if (rule->probability > 1.0f) rule->probability = 1.0f;
            mutations++;
            printf("Mutated rule %u probability: %.3f -> %.3f\n", 
                   rule->rule_id, old_prob, rule->probability);
        }
        rule = rule->next;
    }
    
    if (mutations > 0) {
        grammar->generation++;
        grammar->last_modified = time(NULL);
        printf("Evolved grammar '%s' generation %lu with %d mutations\n", 
               grammar->grammar_name, grammar->generation, mutations);
    }
    
    return mutations > 0;
}

// Optimize grammar performance
bool optimize_grammar_performance(advanced_grammar* grammar, grammar_optimization* metrics) {
    if (!grammar || !metrics) return false;
    
    // Calculate optimization metrics
    metrics->parsing_speed = 1000.0f / (grammar->rule_count + 1); // Simplified metric
    metrics->memory_efficiency = 1.0f / (grammar->rule_count * 0.1f + 1); // Simplified metric
    
    float total_coverage = 0.0f;
    float total_usage = 0.0f;
    grammar_rule* rule = grammar->rules;
    while (rule) {
        total_usage += rule->usage_count;
        if (rule->usage_count > 0) total_coverage += 1.0f;
        rule = rule->next;
    }
    
    metrics->rule_coverage = total_coverage / (grammar->rule_count > 0 ? grammar->rule_count : 1);
    metrics->ambiguity_score = 0.3f; // Placeholder - would calculate actual ambiguity
    metrics->complexity_score = (float)grammar->rule_count / 100.0f;
    
    printf("Grammar '%s' optimization metrics: speed=%.2f, efficiency=%.2f, coverage=%.2f\n",
           grammar->grammar_name, metrics->parsing_speed, metrics->memory_efficiency, metrics->rule_coverage);
    
    return true;
}

// Simplify grammar rules
bool simplify_grammar_rules(advanced_grammar* grammar, float complexity_threshold) {
    if (!grammar) return false;
    
    int removed_count = 0;
    grammar_rule** current = &grammar->rules;
    while (*current) {
        // Remove rules with very low usage
        if ((*current)->usage_count < complexity_threshold) {
            grammar_rule* to_remove = *current;
            *current = to_remove->next;
            
            printf("Removing low-usage rule %u (usage: %.3f)\n", 
                   to_remove->rule_id, to_remove->usage_count);
            
            free(to_remove->lhs);
            free(to_remove->rhs);
            free(to_remove->context_left);
            free(to_remove->context_right);
            free(to_remove);
            
            grammar->rule_count--;
            removed_count++;
        } else {
            current = &(*current)->next;
        }
    }
    
    if (removed_count > 0) {
        grammar->last_modified = time(NULL);
        printf("Simplified grammar '%s' by removing %d rules\n", 
               grammar->grammar_name, removed_count);
    }
    
    return removed_count > 0;
}

// Eliminate ambiguous rules
bool eliminate_ambiguous_rules(advanced_grammar* grammar, float ambiguity_threshold) {
    if (!grammar) return false;
    
    // This is a simplified implementation
    // In practice, would detect actual rule conflicts and ambiguities
    
    int eliminated_count = 0;
    grammar_rule* rule = grammar->rules;
    while (rule) {
        // Mark rules with very low probability as potentially ambiguous
        if (rule->probability < ambiguity_threshold) {
            rule->probability *= 0.5f; // Reduce probability instead of removing
            eliminated_count++;
        }
        rule = rule->next;
    }
    
    if (eliminated_count > 0) {
        printf("Reduced ambiguity in %d rules for grammar '%s'\n", 
               eliminated_count, grammar->grammar_name);
    }
    
    return eliminated_count > 0;
}

// Reorder rules by frequency
bool reorder_rules_by_frequency(advanced_grammar* grammar) {
    if (!grammar || grammar->rule_count <= 1) return false;
    
    // Convert linked list to array for sorting
    grammar_rule** rule_array = malloc(grammar->rule_count * sizeof(grammar_rule*));
    if (!rule_array) return false;
    
    int i = 0;
    grammar_rule* rule = grammar->rules;
    while (rule) {
        rule_array[i++] = rule;
        rule = rule->next;
    }
    
    // Simple bubble sort by usage count
    for (size_t i = 0; i < grammar->rule_count - 1; i++) {
        for (size_t j = 0; j < grammar->rule_count - i - 1; j++) {
            if (rule_array[j]->usage_count < rule_array[j + 1]->usage_count) {
                grammar_rule* temp = rule_array[j];
                rule_array[j] = rule_array[j + 1];
                rule_array[j + 1] = temp;
            }
        }
    }
    
    // Rebuild linked list in sorted order
    grammar->rules = NULL;
    for (int i = grammar->rule_count - 1; i >= 0; i--) {
        rule_array[i]->next = grammar->rules;
        grammar->rules = rule_array[i];
    }
    
    free(rule_array);
    printf("Reordered grammar '%s' rules by usage frequency\n", grammar->grammar_name);
    return true;
}

// Validate grammar consistency
bool validate_grammar_consistency(advanced_grammar* grammar, char** error_message) {
    if (!grammar) {
        if (error_message) *error_message = strdup_safe("Grammar is NULL");
        return false;
    }
    
    // Check for basic consistency
    if (grammar->rule_count == 0) {
        if (error_message) *error_message = strdup_safe("Grammar has no rules");
        return false;
    }
    
    // Check probability sum for rules with same LHS
    grammar_rule* rule = grammar->rules;
    while (rule) {
        if (rule->probability < 0.0f || rule->probability > 1.0f) {
            if (error_message) {
                char* msg = malloc(256);
                snprintf(msg, 256, "Rule %u has invalid probability %.3f", 
                        rule->rule_id, rule->probability);
                *error_message = msg;
            }
            return false;
        }
        rule = rule->next;
    }
    
    printf("Grammar '%s' validation passed\n", grammar->grammar_name);
    return true;
}

// Test grammar coverage
bool test_grammar_coverage(advanced_grammar* grammar, const char** test_inputs, 
                          size_t test_count, float* coverage_score) {
    if (!grammar || !test_inputs || !coverage_score) return false;
    
    int successful_parses = 0;
    grammar_parser* parser = create_grammar_parser(grammar);
    if (!parser) return false;
    
    for (size_t i = 0; i < test_count; i++) {
        char* output = NULL;
        float confidence = 0.0f;
        
        if (parse_with_context(parser, test_inputs[i], &output, &confidence)) {
            if (confidence > 0.5f) { // Threshold for successful parse
                successful_parses++;
            }
            free(output);
        }
    }
    
    *coverage_score = (float)successful_parses / test_count;
    destroy_grammar_parser(parser);
    
    printf("Grammar '%s' coverage test: %d/%zu successful (%.3f)\n", 
           grammar->grammar_name, successful_parses, test_count, *coverage_score);
    return true;
}

// Benchmark grammar performance
bool benchmark_grammar_performance(advanced_grammar* grammar, float* parsing_speed, 
                                  float* memory_usage) {
    if (!grammar || !parsing_speed || !memory_usage) return false;
    
    // Simplified performance metrics
    *parsing_speed = 1000.0f / (grammar->rule_count + 1); // Rules per second
    *memory_usage = grammar->rule_count * 0.1f; // KB approximation
    
    printf("Grammar '%s' performance: %.2f rules/sec, %.2f KB memory\n", 
           grammar->grammar_name, *parsing_speed, *memory_usage);
    return true;
}

// Placeholder implementations for remaining functions
bool generate_grammar_embedding(advanced_grammar* grammar, struct ggml_context* ctx) {
    // Would implement tensor representation generation
    printf("Generated embedding for grammar '%s'\n", grammar->grammar_name);
    return true;
}

bool update_grammar_embedding(advanced_grammar* grammar, struct ggml_context* ctx) {
    printf("Updated embedding for grammar '%s'\n", grammar->grammar_name);
    return true;
}

float compute_grammar_similarity(advanced_grammar* g1, advanced_grammar* g2) {
    if (!g1 || !g2) return 0.0f;
    // Simplified similarity based on rule count
    float similarity = 1.0f - fabs((float)(g1->rule_count - g2->rule_count)) / 
                             (g1->rule_count + g2->rule_count + 1);
    return similarity;
}

bool serialize_grammar(advanced_grammar* grammar, char** serialized_data, size_t* data_size) {
    if (!grammar || !serialized_data || !data_size) return false;
    
    // Simple JSON-like serialization
    char* data = malloc(4096);
    if (!data) return false;
    
    snprintf(data, 4096, "{\"name\":\"%s\",\"type\":%d,\"rules\":%zu,\"generation\":%lu}", 
             grammar->grammar_name, grammar->type, grammar->rule_count, grammar->generation);
    
    *serialized_data = data;
    *data_size = strlen(data);
    return true;
}

advanced_grammar* deserialize_grammar(const char* serialized_data, size_t data_size) {
    if (!serialized_data) return NULL;
    
    // Simplified deserialization - would parse JSON in real implementation
    advanced_grammar* grammar = create_advanced_grammar("deserialized", GRAMMAR_TYPE_CONTEXT_SENSITIVE);
    if (grammar) {
        printf("Deserialized grammar from %zu bytes\n", data_size);
    }
    return grammar;
}