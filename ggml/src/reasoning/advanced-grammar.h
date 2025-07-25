#pragma once

#include "ggml.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations for integration with existing systems
struct cognitive_agent;
struct hypergraph_memory;
struct ggml_tensor;

// Advanced Grammar Types
enum grammar_type {
    GRAMMAR_TYPE_CONTEXT_FREE = 1,
    GRAMMAR_TYPE_CONTEXT_SENSITIVE = 2,
    GRAMMAR_TYPE_RECURSIVE = 3,
    GRAMMAR_TYPE_EVOLVED = 4,
    GRAMMAR_TYPE_COMPOSITE = 5
};

// Grammar rule structure for context-sensitive parsing
typedef struct grammar_rule {
    char* lhs;                      // Left-hand side symbol
    char* rhs;                      // Right-hand side production
    char* context_left;             // Left context for context-sensitive rules
    char* context_right;            // Right context for context-sensitive rules
    float probability;              // Rule probability for stochastic grammars
    float usage_count;              // Usage statistics for evolution
    uint32_t rule_id;              // Unique rule identifier
    struct grammar_rule* next;      // Linked list for chaining
} grammar_rule;

// Grammar composition structure for inheritance
typedef struct grammar_composition {
    struct advanced_grammar* parent_grammar;  // Parent grammar for inheritance
    struct advanced_grammar* child_grammar;   // Child grammar
    float inheritance_weight;                 // Weight for rule inheritance
    uint32_t composition_type;               // Type of composition (union, intersection, etc.)
    struct grammar_composition* next;        // Linked list for multiple compositions
} grammar_composition;

// Context tracking for context-sensitive parsing
typedef struct parsing_context {
    char** context_stack;           // Stack of context symbols
    size_t stack_size;             // Current stack size
    size_t stack_capacity;         // Stack capacity
    uint32_t recursion_depth;      // Current recursion depth
    uint32_t max_recursion_depth;  // Maximum allowed recursion depth
} parsing_context;

// Advanced grammar structure
typedef struct advanced_grammar {
    char* grammar_name;             // Grammar identifier
    enum grammar_type type;         // Grammar type
    grammar_rule* rules;            // Grammar rules (linked list)
    size_t rule_count;             // Number of rules
    parsing_context* context;      // Context for context-sensitive parsing
    grammar_composition* compositions; // Grammar compositions for inheritance
    struct ggml_tensor* embedding;        // Tensor representation of grammar
    float fitness_score;           // Fitness for evolutionary optimization
    uint64_t generation;           // Generation number for evolution
    uint32_t usage_count;          // Usage statistics
    uint64_t created_timestamp;    // When grammar was created
    uint64_t last_modified;        // When grammar was last modified
} advanced_grammar;

// Grammar parser for context-sensitive parsing
typedef struct grammar_parser {
    advanced_grammar* current_grammar;  // Current grammar being used
    parsing_context* parse_context;     // Parsing context state
    char* input_buffer;                 // Input string being parsed
    size_t input_position;             // Current position in input
    size_t input_length;               // Total input length
    uint32_t parse_flags;              // Parsing configuration flags
    float parse_confidence;            // Confidence in parse result
} grammar_parser;

// Grammar optimization metrics
typedef struct grammar_optimization {
    float parsing_speed;           // Average parsing speed
    float memory_efficiency;      // Memory usage efficiency
    float rule_coverage;          // Rule coverage statistics
    float ambiguity_score;        // Grammar ambiguity measurement
    float complexity_score;       // Grammar complexity measurement
} grammar_optimization;

// Function declarations

// Core advanced grammar functions
advanced_grammar* create_advanced_grammar(const char* name, enum grammar_type type);
void destroy_advanced_grammar(advanced_grammar* grammar);
bool add_grammar_rule(advanced_grammar* grammar, const char* lhs, const char* rhs, 
                     const char* context_left, const char* context_right, float probability);
bool remove_grammar_rule(advanced_grammar* grammar, uint32_t rule_id);

// Context-sensitive parsing
grammar_parser* create_grammar_parser(advanced_grammar* grammar);
void destroy_grammar_parser(grammar_parser* parser);
bool parse_with_context(grammar_parser* parser, const char* input, char** output, float* confidence);
bool push_parse_context(parsing_context* context, const char* symbol);
bool pop_parse_context(parsing_context* context, char** symbol);

// Grammar composition and inheritance  
bool compose_grammars(advanced_grammar* parent, advanced_grammar* child, 
                     float inheritance_weight, uint32_t composition_type);
bool inherit_grammar_rules(advanced_grammar* child, advanced_grammar* parent, float threshold);
bool merge_grammar_rules(advanced_grammar* target, advanced_grammar* source, float weight);

// Dynamic grammar modification
bool modify_grammar_rule(advanced_grammar* grammar, uint32_t rule_id, 
                        const char* new_rhs, float new_probability);
bool add_context_sensitivity(advanced_grammar* grammar, uint32_t rule_id,
                           const char* context_left, const char* context_right);
bool evolve_grammar_structure(advanced_grammar* grammar, float mutation_rate);

// Grammar optimization algorithms
bool optimize_grammar_performance(advanced_grammar* grammar, grammar_optimization* metrics);
bool simplify_grammar_rules(advanced_grammar* grammar, float complexity_threshold);
bool eliminate_ambiguous_rules(advanced_grammar* grammar, float ambiguity_threshold);
bool reorder_rules_by_frequency(advanced_grammar* grammar);

// Grammar tensor representation
bool generate_grammar_embedding(advanced_grammar* grammar, struct ggml_context* ctx);
bool update_grammar_embedding(advanced_grammar* grammar, struct ggml_context* ctx);
float compute_grammar_similarity(advanced_grammar* g1, advanced_grammar* g2);

// Integration with cognitive agents
bool install_grammar_in_agent(struct cognitive_agent* agent, advanced_grammar* grammar);
bool share_grammar_between_agents(struct cognitive_agent* source, struct cognitive_agent* target, 
                                 advanced_grammar* grammar);
bool update_agent_grammar_usage(struct cognitive_agent* agent, uint32_t rule_id, float usage_weight);

// Grammar validation and testing
bool validate_grammar_consistency(advanced_grammar* grammar, char** error_message);
bool test_grammar_coverage(advanced_grammar* grammar, const char** test_inputs, 
                          size_t test_count, float* coverage_score);
bool benchmark_grammar_performance(advanced_grammar* grammar, float* parsing_speed, 
                                  float* memory_usage);

// Grammar serialization for sharing
bool serialize_grammar(advanced_grammar* grammar, char** serialized_data, size_t* data_size);
advanced_grammar* deserialize_grammar(const char* serialized_data, size_t data_size);

#ifdef __cplusplus
}
#endif