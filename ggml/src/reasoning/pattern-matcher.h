#pragma once

#include "ggml.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Pattern types
enum pattern_type {
    PATTERN_TYPE_TENSOR = 1,
    PATTERN_TYPE_STRUCTURAL = 2,
    PATTERN_TYPE_SEQUENCE = 3,
    PATTERN_TYPE_TEMPLATE = 4,
    PATTERN_TYPE_MULTIMODAL = 5
};

// Similarity metrics
enum similarity_metric {
    SIMILARITY_COSINE = 1,
    SIMILARITY_EUCLIDEAN = 2,
    SIMILARITY_MANHATTAN = 3,
    SIMILARITY_HAMMING = 4,
    SIMILARITY_JACCARD = 5,
    SIMILARITY_CUSTOM = 6
};

// Pattern structure
struct pattern {
    enum pattern_type type;
    struct ggml_tensor* data;          // Pattern data as tensor
    char* name;                        // Pattern identifier
    float* features;                   // Extracted feature vector
    size_t feature_count;              // Number of features
    
    // Pattern metadata
    uint64_t pattern_id;               // Unique identifier
    float confidence;                  // Pattern confidence score
    uint64_t creation_time;            // When pattern was created
    uint64_t last_matched;             // Last time pattern was matched
    uint32_t match_count;              // Number of times matched
    
    // Structural information (for hypergraph patterns)
    void** nodes;                      // Pattern nodes (generic pointers)
    void** edges;                      // Pattern edges
    size_t node_count;                 // Number of nodes
    size_t edge_count;                 // Number of edges
    
    // Temporal information (for sequence patterns)
    float* timestamps;                 // Temporal sequence data
    size_t sequence_length;            // Length of sequence
    float temporal_weight;             // Importance of temporal ordering
};

// Pattern template for matching
struct pattern_template {
    char* template_name;               // Template identifier
    enum pattern_type expected_type;   // Expected pattern type
    float* feature_weights;            // Feature importance weights
    size_t weight_count;               // Number of weights
    
    // Matching criteria
    float min_similarity;              // Minimum similarity threshold
    float max_dissimilarity;           // Maximum allowed dissimilarity
    bool require_structural_match;     // Whether structure must match
    bool require_temporal_order;       // Whether temporal order matters
    
    // Template constraints
    size_t min_nodes;                  // Minimum number of nodes
    size_t max_nodes;                  // Maximum number of nodes
    float complexity_penalty;          // Penalty for complex patterns
};

// Pattern match result
struct pattern_match {
    struct pattern* source_pattern;    // Source pattern
    struct pattern* target_pattern;    // Target pattern or data
    float similarity_score;            // Overall similarity
    float confidence_score;            // Match confidence
    enum similarity_metric metric_used; // Similarity metric used
    
    // Detailed match information
    float* feature_similarities;       // Per-feature similarity scores
    size_t similarity_count;           // Number of similarity scores
    
    // Structural match info
    size_t matched_nodes;              // Number of matched nodes
    size_t matched_edges;              // Number of matched edges
    float structural_similarity;       // Structural similarity score
    
    // Temporal match info
    float temporal_similarity;         // Temporal alignment score
    float* temporal_offsets;           // Time offsets for alignment
    size_t offset_count;               // Number of offsets
    
    // Match metadata
    uint64_t match_time;               // When match was computed
    bool is_exact_match;               // Whether this is an exact match
    bool is_fuzzy_match;               // Whether this is a fuzzy match
};

// Pattern matcher engine
struct pattern_matcher {
    struct ggml_context* ctx;
    
    // Pattern storage
    struct pattern** patterns;         // Array of stored patterns
    size_t pattern_count;              // Number of stored patterns
    size_t pattern_capacity;           // Pattern storage capacity
    
    // Template storage
    struct pattern_template** templates; // Array of pattern templates
    size_t template_count;             // Number of templates
    size_t template_capacity;          // Template storage capacity
    
    // Matching configuration
    enum similarity_metric default_metric; // Default similarity metric
    float default_threshold;           // Default similarity threshold
    bool enable_fuzzy_matching;        // Whether to use fuzzy matching
    bool enable_analogy_detection;     // Whether to detect analogies
    
    // Performance settings
    size_t max_comparison_time_ns;     // Maximum time per comparison
    size_t cache_size;                 // Size of similarity cache
    bool use_approximation;            // Whether to use approximations
    
    // Statistics
    uint64_t patterns_matched;         // Total patterns matched
    uint64_t comparisons_made;         // Total comparisons made
    uint64_t cache_hits;               // Cache hit count
    float average_similarity;          // Average similarity found
    float average_match_time_ns;       // Average matching time
};

// Core pattern matcher functions
struct pattern_matcher* pattern_matcher_create(struct ggml_context* ctx);
void pattern_matcher_destroy(struct pattern_matcher* matcher);

// Pattern management
struct pattern* pattern_create(enum pattern_type type, struct ggml_tensor* data, const char* name);
void pattern_destroy(struct pattern* pattern);
int pattern_matcher_add_pattern(struct pattern_matcher* matcher, struct pattern* pattern);
struct pattern* pattern_matcher_find_pattern(struct pattern_matcher* matcher, const char* name);
int pattern_matcher_remove_pattern(struct pattern_matcher* matcher, const char* name);

// Feature extraction
int pattern_extract_features(struct pattern* pattern);
int pattern_extract_tensor_features(struct pattern* pattern);
int pattern_extract_structural_features(struct pattern* pattern);
int pattern_extract_temporal_features(struct pattern* pattern);

// Similarity computation
float pattern_compute_similarity(struct pattern* pattern1, struct pattern* pattern2, enum similarity_metric metric);
float pattern_cosine_similarity(struct pattern* pattern1, struct pattern* pattern2);
float pattern_euclidean_similarity(struct pattern* pattern1, struct pattern* pattern2);
float pattern_structural_similarity(struct pattern* pattern1, struct pattern* pattern2);
float pattern_temporal_similarity(struct pattern* pattern1, struct pattern* pattern2);

// Pattern matching
struct pattern_match* pattern_matcher_match(struct pattern_matcher* matcher, struct pattern* query_pattern);
struct pattern_match** pattern_matcher_match_all(struct pattern_matcher* matcher, struct pattern* query_pattern, 
                                                  float threshold, size_t* match_count);
struct pattern_match* pattern_matcher_find_best_match(struct pattern_matcher* matcher, struct pattern* query_pattern);

// Fuzzy matching
struct pattern_match** pattern_matcher_fuzzy_match(struct pattern_matcher* matcher, struct pattern* query_pattern,
                                                   float min_threshold, float max_threshold, size_t* match_count);
bool pattern_matcher_is_fuzzy_match(struct pattern_match* match, float threshold);

// Template matching
struct pattern_template* pattern_template_create(const char* name, enum pattern_type type);
void pattern_template_destroy(struct pattern_template* template);
int pattern_matcher_add_template(struct pattern_matcher* matcher, struct pattern_template* template);
struct pattern_match** pattern_matcher_match_template(struct pattern_matcher* matcher, 
                                                      struct pattern_template* template,
                                                      struct pattern* query_pattern, size_t* match_count);

// Analogy detection
struct pattern_match** pattern_matcher_detect_analogies(struct pattern_matcher* matcher,
                                                        struct pattern* source, struct pattern* target,
                                                        size_t* analogy_count);
float pattern_compute_analogy_strength(struct pattern* pattern_a, struct pattern* pattern_b,
                                       struct pattern* pattern_c, struct pattern* pattern_d);

// Cross-modal pattern matching
struct pattern_match* pattern_matcher_cross_modal_match(struct pattern_matcher* matcher,
                                                       struct pattern* text_pattern,
                                                       struct pattern* embedding_pattern,
                                                       struct pattern* structure_pattern);

// Pattern composition and decomposition
struct pattern* pattern_compose(struct pattern* pattern1, struct pattern* pattern2, float weight1, float weight2);
struct pattern** pattern_decompose(struct pattern* complex_pattern, size_t target_components, size_t* component_count);

// Configuration
void pattern_matcher_set_threshold(struct pattern_matcher* matcher, float threshold);
void pattern_matcher_set_metric(struct pattern_matcher* matcher, enum similarity_metric metric);
void pattern_matcher_enable_fuzzy_matching(struct pattern_matcher* matcher, bool enable);
void pattern_matcher_enable_analogy_detection(struct pattern_matcher* matcher, bool enable);

// Performance and statistics
void pattern_matcher_print_stats(struct pattern_matcher* matcher);
float pattern_matcher_get_accuracy(struct pattern_matcher* matcher);
void pattern_matcher_reset_stats(struct pattern_matcher* matcher);

// Integration with cognitive agents
int pattern_matcher_integrate_with_memory(struct pattern_matcher* matcher, void* hypergraph_memory);
int pattern_matcher_integrate_with_reasoning(struct pattern_matcher* matcher, void* reasoning_engine);

// Utility functions
void pattern_match_destroy(struct pattern_match* match);
uint64_t pattern_get_timestamp(void);
float pattern_random_float(void);

#ifdef __cplusplus
}
#endif