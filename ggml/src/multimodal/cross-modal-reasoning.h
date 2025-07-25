#pragma once

#include "multimodal-core.h"
#include "advanced-grammar.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct cognitive_agent;
struct pattern_matcher;

// Cross-modal reasoning types
enum reasoning_type {
    REASONING_ANALOGICAL = 1,
    REASONING_CAUSAL = 2,
    REASONING_COMPOSITIONAL = 3,
    REASONING_TEMPORAL = 4,
    REASONING_SPATIAL = 5
};

// Cross-modal analogy structure
typedef struct cross_modal_analogy {
    multimodal_tensor* source_modality;    // Source modality for analogy
    multimodal_tensor* target_modality;    // Target modality for analogy
    float* correspondence_weights;         // Correspondence strengths
    char* analogy_description;             // Textual description of analogy
    float confidence_score;                // Confidence in analogy
    enum reasoning_type reasoning_type;    // Type of reasoning used
} cross_modal_analogy;

// Multi-modal pattern for cross-modal matching
typedef struct multimodal_pattern {
    multimodal_tensor** modal_components;  // Components from different modalities
    size_t component_count;                // Number of components
    float* component_weights;              // Weights for each component
    char* pattern_name;                    // Name/identifier for pattern
    float occurrence_frequency;            // How often pattern occurs
    uint64_t last_matched;                 // When pattern was last matched
} multimodal_pattern;

// Cross-modal reasoning engine
typedef struct cross_modal_reasoning_engine {
    multimodal_processor* mm_processor;     // Multi-modal processor
    struct pattern_matcher* pattern_matcher; // Pattern matching engine
    advanced_grammar* reasoning_grammar;    // Grammar for reasoning expressions
    cross_modal_analogy** analogies;       // Stored analogies
    size_t analogy_count;                   // Number of analogies
    size_t analogy_capacity;                // Analogy storage capacity
    multimodal_pattern** patterns;          // Multi-modal patterns
    size_t pattern_count;                   // Number of patterns
    size_t pattern_capacity;                // Pattern storage capacity
    float reasoning_threshold;              // Minimum confidence for reasoning
    bool learning_enabled;                  // Whether to learn new patterns
} cross_modal_reasoning_engine;

// Temporal reasoning context
typedef struct temporal_reasoning_context {
    multimodal_tensor** temporal_sequence; // Sequence of multi-modal states
    size_t sequence_length;                // Length of temporal sequence
    uint64_t* timestamps;                  // Timestamps for each state
    float* transition_probabilities;       // Probabilities between states
    char* temporal_pattern;                // Detected temporal pattern
} temporal_reasoning_context;

// Spatial reasoning context
typedef struct spatial_reasoning_context {
    multimodal_tensor** spatial_objects;   // Objects in spatial arrangement
    size_t object_count;                   // Number of objects
    float** spatial_relations;            // Spatial relationship matrix
    char** relation_types;                 // Types of spatial relations
    float* object_positions;               // 3D positions of objects
} spatial_reasoning_context;

// Function declarations

// Cross-modal reasoning engine management
cross_modal_reasoning_engine* create_cross_modal_reasoning_engine(multimodal_processor* mm_processor);
void destroy_cross_modal_reasoning_engine(cross_modal_reasoning_engine* engine);
bool configure_reasoning_parameters(cross_modal_reasoning_engine* engine, float threshold, bool enable_learning);

// Multi-modal pattern matching
bool detect_multimodal_patterns(cross_modal_reasoning_engine* engine, multimodal_tensor** inputs, size_t input_count, multimodal_pattern*** detected_patterns, size_t* pattern_count);
bool match_cross_modal_features(multimodal_tensor* tensor1, multimodal_tensor* tensor2, float* match_score, float** correspondence_map);
bool learn_multimodal_pattern(cross_modal_reasoning_engine* engine, multimodal_tensor** components, size_t component_count, const char* pattern_name);
bool update_pattern_statistics(cross_modal_reasoning_engine* engine, multimodal_pattern* pattern, bool matched);

// Cross-modal analogical reasoning
bool detect_cross_modal_analogies(cross_modal_reasoning_engine* engine, multimodal_tensor** inputs, size_t input_count, cross_modal_analogy*** analogies, size_t* analogy_count);
bool create_cross_modal_analogy(multimodal_tensor* source, multimodal_tensor* target, enum reasoning_type type, cross_modal_analogy** analogy);
bool apply_analogy_to_reasoning(cross_modal_reasoning_engine* engine, cross_modal_analogy* analogy, multimodal_tensor* query, multimodal_tensor** reasoning_result);
bool validate_analogy_consistency(cross_modal_analogy* analogy, float* consistency_score);

// Cross-modal causal reasoning
bool infer_cross_modal_causality(cross_modal_reasoning_engine* engine, multimodal_tensor* cause_modality, multimodal_tensor* effect_modality, float* causality_strength);
bool build_causal_model(cross_modal_reasoning_engine* engine, multimodal_tensor** observations, size_t observation_count, void** causal_model);
bool predict_cross_modal_effects(cross_modal_reasoning_engine* engine, void* causal_model, multimodal_tensor* intervention, multimodal_tensor** predicted_effects, size_t* effect_count);

// Compositional reasoning
bool decompose_multimodal_concept(cross_modal_reasoning_engine* engine, multimodal_tensor* complex_concept, multimodal_tensor*** components, size_t* component_count);
bool compose_multimodal_concept(cross_modal_reasoning_engine* engine, multimodal_tensor** components, size_t component_count, multimodal_tensor** composed_concept);
bool reason_about_composition(cross_modal_reasoning_engine* engine, multimodal_tensor* whole, multimodal_tensor* part, float* composition_strength);

// Temporal reasoning
temporal_reasoning_context* create_temporal_reasoning_context(size_t max_sequence_length);
void destroy_temporal_reasoning_context(temporal_reasoning_context* context);
bool add_temporal_state(temporal_reasoning_context* context, multimodal_tensor* state, uint64_t timestamp);
bool detect_temporal_patterns(cross_modal_reasoning_engine* engine, temporal_reasoning_context* context, char** detected_pattern, float* pattern_confidence);
bool predict_temporal_progression(cross_modal_reasoning_engine* engine, temporal_reasoning_context* context, multimodal_tensor** next_states, size_t* prediction_count);

// Spatial reasoning
spatial_reasoning_context* create_spatial_reasoning_context(size_t max_objects);
void destroy_spatial_reasoning_context(spatial_reasoning_context* context);
bool add_spatial_object(spatial_reasoning_context* context, multimodal_tensor* object, float* position);
bool compute_spatial_relations(spatial_reasoning_context* context);
bool reason_about_spatial_arrangement(cross_modal_reasoning_engine* engine, spatial_reasoning_context* context, const char* query, char** reasoning_result, float* confidence);

// Multi-modal memory integration for reasoning
bool integrate_reasoning_with_memory(cross_modal_reasoning_engine* engine, multimodal_memory* memory);
bool store_reasoning_result(cross_modal_reasoning_engine* engine, multimodal_tensor* input, multimodal_tensor* result, enum reasoning_type type);
bool retrieve_similar_reasoning_cases(cross_modal_reasoning_engine* engine, multimodal_tensor* query, cross_modal_analogy*** similar_cases, size_t* case_count);

// Advanced reasoning capabilities
bool perform_counterfactual_reasoning(cross_modal_reasoning_engine* engine, multimodal_tensor* factual_scenario, multimodal_tensor* counterfactual_intervention, multimodal_tensor** counterfactual_result);
bool reason_about_modal_interactions(cross_modal_reasoning_engine* engine, multimodal_tensor** modalities, size_t modality_count, char** interaction_description, float* interaction_strength);
bool transfer_reasoning_across_modalities(cross_modal_reasoning_engine* engine, multimodal_tensor* source_reasoning, enum modality_type target_modality, multimodal_tensor** transferred_reasoning);

// Grammar-guided reasoning
bool parse_reasoning_query(cross_modal_reasoning_engine* engine, const char* query, enum reasoning_type* type, multimodal_tensor** query_tensor);
bool generate_reasoning_explanation(cross_modal_reasoning_engine* engine, multimodal_tensor* input, multimodal_tensor* output, char** explanation);
bool validate_reasoning_with_grammar(cross_modal_reasoning_engine* engine, multimodal_tensor* reasoning_chain, bool* is_valid, char** validation_errors);

// Learning and adaptation
bool learn_from_reasoning_feedback(cross_modal_reasoning_engine* engine, multimodal_tensor* input, multimodal_tensor* expected_output, multimodal_tensor* actual_output);
bool adapt_reasoning_strategies(cross_modal_reasoning_engine* engine, float* strategy_weights, size_t strategy_count);
bool update_analogy_database(cross_modal_reasoning_engine* engine, cross_modal_analogy* new_analogy);

// Performance optimization
bool optimize_reasoning_performance(cross_modal_reasoning_engine* engine);
bool cache_frequent_reasoning_patterns(cross_modal_reasoning_engine* engine);
bool prune_low_confidence_analogies(cross_modal_reasoning_engine* engine, float confidence_threshold);

// Integration with cognitive agents
bool install_cross_modal_reasoning(struct cognitive_agent* agent, cross_modal_reasoning_engine* engine);
bool process_cross_modal_reasoning_request(struct cognitive_agent* agent, multimodal_tensor* query, enum reasoning_type type, multimodal_tensor** result, char** explanation);
bool share_reasoning_knowledge(struct cognitive_agent** agents, size_t agent_count, cross_modal_analogy** analogies, size_t analogy_count);

// Evaluation and metrics
bool evaluate_reasoning_accuracy(cross_modal_reasoning_engine* engine, multimodal_tensor** test_inputs, multimodal_tensor** expected_outputs, size_t test_count, float* accuracy);
bool measure_reasoning_consistency(cross_modal_reasoning_engine* engine, multimodal_tensor* query, size_t repetitions, float* consistency_score);
bool assess_reasoning_generalization(cross_modal_reasoning_engine* engine, multimodal_tensor** novel_inputs, size_t input_count, float* generalization_score);

// Serialization and sharing
bool serialize_reasoning_state(cross_modal_reasoning_engine* engine, uint8_t** serialized_data, size_t* data_size);
cross_modal_reasoning_engine* deserialize_reasoning_state(uint8_t* serialized_data, size_t data_size, multimodal_processor* mm_processor);
bool export_analogies_to_file(cross_modal_reasoning_engine* engine, const char* filename);
bool import_analogies_from_file(cross_modal_reasoning_engine* engine, const char* filename);

#ifdef __cplusplus
}
#endif