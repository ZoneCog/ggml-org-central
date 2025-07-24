#include "pattern-matcher.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

// Get high-resolution timestamp
uint64_t pattern_get_timestamp(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

// Generate random float [0, 1)
float pattern_random_float(void) {
    return (float)rand() / (float)RAND_MAX;
}

// Create pattern matcher
struct pattern_matcher* pattern_matcher_create(struct ggml_context* ctx) {
    struct pattern_matcher* matcher = malloc(sizeof(struct pattern_matcher));
    if (!matcher) return NULL;
    
    matcher->ctx = ctx;
    matcher->pattern_capacity = 1000;
    matcher->patterns = malloc(sizeof(struct pattern*) * matcher->pattern_capacity);
    matcher->pattern_count = 0;
    
    matcher->template_capacity = 100;
    matcher->templates = malloc(sizeof(struct pattern_template*) * matcher->template_capacity);
    matcher->template_count = 0;
    
    // Default configuration
    matcher->default_metric = SIMILARITY_COSINE;
    matcher->default_threshold = 0.7f;
    matcher->enable_fuzzy_matching = true;
    matcher->enable_analogy_detection = true;
    
    // Performance settings
    matcher->max_comparison_time_ns = 1000000; // 1ms
    matcher->cache_size = 10000;
    matcher->use_approximation = false;
    
    // Initialize statistics
    matcher->patterns_matched = 0;
    matcher->comparisons_made = 0;
    matcher->cache_hits = 0;
    matcher->average_similarity = 0.0f;
    matcher->average_match_time_ns = 0.0f;
    
    return matcher;
}

// Destroy pattern matcher
void pattern_matcher_destroy(struct pattern_matcher* matcher) {
    if (!matcher) return;
    
    // Clean up patterns
    for (size_t i = 0; i < matcher->pattern_count; i++) {
        pattern_destroy(matcher->patterns[i]);
    }
    free(matcher->patterns);
    
    // Clean up templates
    for (size_t i = 0; i < matcher->template_count; i++) {
        pattern_template_destroy(matcher->templates[i]);
    }
    free(matcher->templates);
    
    free(matcher);
}

// Create pattern
struct pattern* pattern_create(enum pattern_type type, struct ggml_tensor* data, const char* name) {
    struct pattern* pattern = malloc(sizeof(struct pattern));
    if (!pattern) return NULL;
    
    pattern->type = type;
    pattern->data = data;
    
    if (name) {
        pattern->name = malloc(strlen(name) + 1);
        strcpy(pattern->name, name);
    } else {
        pattern->name = NULL;
    }
    
    // Initialize feature vector
    pattern->feature_count = 128; // Default feature vector size
    pattern->features = malloc(sizeof(float) * pattern->feature_count);
    
    // Initialize metadata
    static uint64_t next_id = 1;
    pattern->pattern_id = next_id++;
    pattern->confidence = 1.0f;
    pattern->creation_time = pattern_get_timestamp();
    pattern->last_matched = 0;
    pattern->match_count = 0;
    
    // Initialize structural information
    pattern->nodes = NULL;
    pattern->edges = NULL;
    pattern->node_count = 0;
    pattern->edge_count = 0;
    
    // Initialize temporal information
    pattern->timestamps = NULL;
    pattern->sequence_length = 0;
    pattern->temporal_weight = 1.0f;
    
    return pattern;
}

// Destroy pattern
void pattern_destroy(struct pattern* pattern) {
    if (!pattern) return;
    
    free(pattern->name);
    free(pattern->features);
    free(pattern->nodes);
    free(pattern->edges);
    free(pattern->timestamps);
    free(pattern);
}

// Add pattern to matcher
int pattern_matcher_add_pattern(struct pattern_matcher* matcher, struct pattern* pattern) {
    if (!matcher || !pattern) return -1;
    
    if (matcher->pattern_count >= matcher->pattern_capacity) {
        matcher->pattern_capacity *= 2;
        matcher->patterns = realloc(matcher->patterns, 
                                   sizeof(struct pattern*) * matcher->pattern_capacity);
    }
    
    // Extract features for the pattern
    pattern_extract_features(pattern);
    
    matcher->patterns[matcher->pattern_count++] = pattern;
    return 0;
}

// Find pattern by name
struct pattern* pattern_matcher_find_pattern(struct pattern_matcher* matcher, const char* name) {
    if (!matcher || !name) return NULL;
    
    for (size_t i = 0; i < matcher->pattern_count; i++) {
        if (matcher->patterns[i]->name && strcmp(matcher->patterns[i]->name, name) == 0) {
            return matcher->patterns[i];
        }
    }
    return NULL;
}

// Extract features from pattern
int pattern_extract_features(struct pattern* pattern) {
    if (!pattern) return -1;
    
    switch (pattern->type) {
        case PATTERN_TYPE_TENSOR:
            return pattern_extract_tensor_features(pattern);
        case PATTERN_TYPE_STRUCTURAL:
            return pattern_extract_structural_features(pattern);
        case PATTERN_TYPE_SEQUENCE:
            return pattern_extract_temporal_features(pattern);
        default:
            return pattern_extract_tensor_features(pattern);
    }
}

// Extract tensor features
int pattern_extract_tensor_features(struct pattern* pattern) {
    if (!pattern || !pattern->data) return -1;
    
    // Simple feature extraction: use tensor data directly or compute statistics
    size_t tensor_size = ggml_nelements(pattern->data);
    float* data = (float*)pattern->data->data;
    
    if (!data) return -1;
    
    // Ensure we have enough features
    if (pattern->feature_count < 8) {
        pattern->feature_count = 8;
        pattern->features = realloc(pattern->features, sizeof(float) * pattern->feature_count);
    }
    
    // Compute basic statistical features
    float sum = 0.0f, mean = 0.0f, variance = 0.0f;
    float min_val = data[0], max_val = data[0];
    
    for (size_t i = 0; i < tensor_size; i++) {
        sum += data[i];
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }
    
    mean = sum / tensor_size;
    
    for (size_t i = 0; i < tensor_size; i++) {
        float diff = data[i] - mean;
        variance += diff * diff;
    }
    variance /= tensor_size;
    
    // Store statistical features
    pattern->features[0] = mean;
    pattern->features[1] = sqrtf(variance); // Standard deviation
    pattern->features[2] = min_val;
    pattern->features[3] = max_val;
    pattern->features[4] = max_val - min_val; // Range
    pattern->features[5] = sum;
    pattern->features[6] = (float)tensor_size; // Size
    pattern->features[7] = sum / (max_val - min_val + 1e-6f); // Normalized sum
    
    return 0;
}

// Extract structural features
int pattern_extract_structural_features(struct pattern* pattern) {
    if (!pattern) return -1;
    
    // For structural patterns, extract graph-theoretic features
    pattern->features[0] = (float)pattern->node_count;
    pattern->features[1] = (float)pattern->edge_count;
    pattern->features[2] = pattern->edge_count > 0 ? 
                          (float)pattern->edge_count / pattern->node_count : 0.0f; // Edge density
    pattern->features[3] = pattern->confidence;
    
    return 0;
}

// Extract temporal features
int pattern_extract_temporal_features(struct pattern* pattern) {
    if (!pattern || !pattern->timestamps) return -1;
    
    // Compute temporal statistics
    float duration = 0.0f;
    if (pattern->sequence_length > 1) {
        duration = pattern->timestamps[pattern->sequence_length - 1] - pattern->timestamps[0];
    }
    
    pattern->features[0] = (float)pattern->sequence_length;
    pattern->features[1] = duration;
    pattern->features[2] = pattern->temporal_weight;
    pattern->features[3] = duration > 0 ? pattern->sequence_length / duration : 0.0f; // Frequency
    
    return 0;
}

// Compute cosine similarity
float pattern_cosine_similarity(struct pattern* pattern1, struct pattern* pattern2) {
    if (!pattern1 || !pattern2 || !pattern1->features || !pattern2->features) return 0.0f;
    
    size_t feature_count = (pattern1->feature_count < pattern2->feature_count) ?
                          pattern1->feature_count : pattern2->feature_count;
    
    float dot_product = 0.0f;
    float norm1 = 0.0f, norm2 = 0.0f;
    
    for (size_t i = 0; i < feature_count; i++) {
        dot_product += pattern1->features[i] * pattern2->features[i];
        norm1 += pattern1->features[i] * pattern1->features[i];
        norm2 += pattern2->features[i] * pattern2->features[i];
    }
    
    norm1 = sqrtf(norm1);
    norm2 = sqrtf(norm2);
    
    if (norm1 < 1e-6f || norm2 < 1e-6f) return 0.0f;
    
    return dot_product / (norm1 * norm2);
}

// Compute Euclidean similarity (converted to similarity score)
float pattern_euclidean_similarity(struct pattern* pattern1, struct pattern* pattern2) {
    if (!pattern1 || !pattern2 || !pattern1->features || !pattern2->features) return 0.0f;
    
    size_t feature_count = (pattern1->feature_count < pattern2->feature_count) ?
                          pattern1->feature_count : pattern2->feature_count;
    
    float distance = 0.0f;
    
    for (size_t i = 0; i < feature_count; i++) {
        float diff = pattern1->features[i] - pattern2->features[i];
        distance += diff * diff;
    }
    
    distance = sqrtf(distance);
    
    // Convert distance to similarity (0 distance = 1 similarity)
    return 1.0f / (1.0f + distance);
}

// Compute structural similarity
float pattern_structural_similarity(struct pattern* pattern1, struct pattern* pattern2) {
    if (!pattern1 || !pattern2) return 0.0f;
    
    // Simple structural similarity based on node and edge counts
    float node_sim = 1.0f - fabsf((float)pattern1->node_count - pattern2->node_count) / 
                            fmaxf((float)pattern1->node_count, pattern2->node_count);
    float edge_sim = 1.0f - fabsf((float)pattern1->edge_count - pattern2->edge_count) / 
                            fmaxf((float)pattern1->edge_count, pattern2->edge_count);
    
    return (node_sim + edge_sim) / 2.0f;
}

// Compute temporal similarity
float pattern_temporal_similarity(struct pattern* pattern1, struct pattern* pattern2) {
    if (!pattern1 || !pattern2) return 0.0f;
    
    // Simple temporal similarity based on sequence length
    float length_sim = 1.0f - fabsf((float)pattern1->sequence_length - pattern2->sequence_length) / 
                              fmaxf((float)pattern1->sequence_length, pattern2->sequence_length);
    
    return length_sim * fminf(pattern1->temporal_weight, pattern2->temporal_weight);
}

// Compute similarity using specified metric
float pattern_compute_similarity(struct pattern* pattern1, struct pattern* pattern2, enum similarity_metric metric) {
    if (!pattern1 || !pattern2) return 0.0f;
    
    switch (metric) {
        case SIMILARITY_COSINE:
            return pattern_cosine_similarity(pattern1, pattern2);
        case SIMILARITY_EUCLIDEAN:
            return pattern_euclidean_similarity(pattern1, pattern2);
        case SIMILARITY_MANHATTAN:
            // Implement Manhattan distance similarity
            return pattern_euclidean_similarity(pattern1, pattern2) * 0.9f; // Approximation
        case SIMILARITY_HAMMING:
            // Implement Hamming distance similarity for discrete patterns
            return pattern_structural_similarity(pattern1, pattern2);
        case SIMILARITY_JACCARD:
            // Implement Jaccard similarity for set-based patterns
            return pattern_structural_similarity(pattern1, pattern2) * 0.8f; // Approximation
        default:
            return pattern_cosine_similarity(pattern1, pattern2);
    }
}

// Create pattern match result
struct pattern_match* pattern_match_create(struct pattern* source, struct pattern* target, 
                                          float similarity, enum similarity_metric metric) {
    struct pattern_match* match = malloc(sizeof(struct pattern_match));
    if (!match) return NULL;
    
    match->source_pattern = source;
    match->target_pattern = target;
    match->similarity_score = similarity;
    match->confidence_score = similarity; // Simple confidence = similarity
    match->metric_used = metric;
    
    // Initialize detailed match information
    match->feature_similarities = NULL;
    match->similarity_count = 0;
    match->matched_nodes = 0;
    match->matched_edges = 0;
    match->structural_similarity = 0.0f;
    match->temporal_similarity = 0.0f;
    match->temporal_offsets = NULL;
    match->offset_count = 0;
    
    // Set match metadata
    match->match_time = pattern_get_timestamp();
    match->is_exact_match = (similarity > 0.99f);
    match->is_fuzzy_match = (similarity > 0.5f && similarity < 0.99f);
    
    return match;
}

// Destroy pattern match
void pattern_match_destroy(struct pattern_match* match) {
    if (!match) return;
    
    free(match->feature_similarities);
    free(match->temporal_offsets);
    free(match);
}

// Find best match for a pattern
struct pattern_match* pattern_matcher_find_best_match(struct pattern_matcher* matcher, struct pattern* query_pattern) {
    if (!matcher || !query_pattern) return NULL;
    
    struct pattern_match* best_match = NULL;
    float best_similarity = -1.0f;
    
    uint64_t start_time = pattern_get_timestamp();
    
    for (size_t i = 0; i < matcher->pattern_count; i++) {
        float similarity = pattern_compute_similarity(query_pattern, matcher->patterns[i], 
                                                     matcher->default_metric);
        matcher->comparisons_made++;
        
        if (similarity > best_similarity) {
            if (best_match) {
                pattern_match_destroy(best_match);
            }
            best_match = pattern_match_create(query_pattern, matcher->patterns[i], 
                                            similarity, matcher->default_metric);
            best_similarity = similarity;
        }
        
        // Update target pattern's match statistics
        matcher->patterns[i]->last_matched = pattern_get_timestamp();
        if (similarity > matcher->default_threshold) {
            matcher->patterns[i]->match_count++;
        }
    }
    
    // Update statistics
    if (best_match) {
        matcher->patterns_matched++;
        matcher->average_similarity = (matcher->average_similarity * (matcher->patterns_matched - 1) + 
                                      best_similarity) / matcher->patterns_matched;
    }
    
    uint64_t match_time = pattern_get_timestamp() - start_time;
    matcher->average_match_time_ns = (matcher->average_match_time_ns * (matcher->patterns_matched - 1) + 
                                     match_time) / matcher->patterns_matched;
    
    return best_match;
}

// Find all matches above threshold
struct pattern_match** pattern_matcher_match_all(struct pattern_matcher* matcher, struct pattern* query_pattern, 
                                                 float threshold, size_t* match_count) {
    if (!matcher || !query_pattern || !match_count) return NULL;
    
    *match_count = 0;
    struct pattern_match** matches = malloc(sizeof(struct pattern_match*) * matcher->pattern_count);
    
    for (size_t i = 0; i < matcher->pattern_count; i++) {
        float similarity = pattern_compute_similarity(query_pattern, matcher->patterns[i], 
                                                     matcher->default_metric);
        matcher->comparisons_made++;
        
        if (similarity >= threshold) {
            matches[*match_count] = pattern_match_create(query_pattern, matcher->patterns[i], 
                                                        similarity, matcher->default_metric);
            (*match_count)++;
        }
    }
    
    if (*match_count == 0) {
        free(matches);
        return NULL;
    }
    
    // Resize to actual count
    matches = realloc(matches, sizeof(struct pattern_match*) * (*match_count));
    
    matcher->patterns_matched += *match_count;
    return matches;
}

// Create pattern template
struct pattern_template* pattern_template_create(const char* name, enum pattern_type type) {
    struct pattern_template* template = malloc(sizeof(struct pattern_template));
    if (!template) return NULL;
    
    if (name) {
        template->template_name = malloc(strlen(name) + 1);
        strcpy(template->template_name, name);
    } else {
        template->template_name = NULL;
    }
    
    template->expected_type = type;
    template->weight_count = 128; // Default
    template->feature_weights = malloc(sizeof(float) * template->weight_count);
    
    // Initialize with uniform weights
    for (size_t i = 0; i < template->weight_count; i++) {
        template->feature_weights[i] = 1.0f;
    }
    
    // Default matching criteria
    template->min_similarity = 0.7f;
    template->max_dissimilarity = 0.3f;
    template->require_structural_match = false;
    template->require_temporal_order = false;
    
    // Default constraints
    template->min_nodes = 0;
    template->max_nodes = 1000;
    template->complexity_penalty = 0.01f;
    
    return template;
}

// Destroy pattern template
void pattern_template_destroy(struct pattern_template* template) {
    if (!template) return;
    
    free(template->template_name);
    free(template->feature_weights);
    free(template);
}

// Configuration functions
void pattern_matcher_set_threshold(struct pattern_matcher* matcher, float threshold) {
    if (matcher) {
        matcher->default_threshold = threshold;
    }
}

void pattern_matcher_set_metric(struct pattern_matcher* matcher, enum similarity_metric metric) {
    if (matcher) {
        matcher->default_metric = metric;
    }
}

void pattern_matcher_enable_fuzzy_matching(struct pattern_matcher* matcher, bool enable) {
    if (matcher) {
        matcher->enable_fuzzy_matching = enable;
    }
}

void pattern_matcher_enable_analogy_detection(struct pattern_matcher* matcher, bool enable) {
    if (matcher) {
        matcher->enable_analogy_detection = enable;
    }
}

// Print statistics
void pattern_matcher_print_stats(struct pattern_matcher* matcher) {
    if (!matcher) {
        printf("Pattern matcher not initialized\n");
        return;
    }
    
    printf("Pattern Matcher Statistics:\n");
    printf("  Stored patterns: %zu\n", matcher->pattern_count);
    printf("  Patterns matched: %lu\n", matcher->patterns_matched);
    printf("  Comparisons made: %lu\n", matcher->comparisons_made);
    printf("  Cache hits: %lu\n", matcher->cache_hits);
    printf("  Average similarity: %.4f\n", matcher->average_similarity);
    printf("  Average match time: %.2f μs\n", matcher->average_match_time_ns / 1000.0f);
    printf("  Default threshold: %.3f\n", matcher->default_threshold);
    printf("  Fuzzy matching: %s\n", matcher->enable_fuzzy_matching ? "Enabled" : "Disabled");
    printf("  Analogy detection: %s\n", matcher->enable_analogy_detection ? "Enabled" : "Disabled");
}

// Get accuracy (simplified)
float pattern_matcher_get_accuracy(struct pattern_matcher* matcher) {
    if (!matcher || matcher->patterns_matched == 0) return 0.0f;
    
    return matcher->average_similarity;
}

// Reset statistics
void pattern_matcher_reset_stats(struct pattern_matcher* matcher) {
    if (!matcher) return;
    
    matcher->patterns_matched = 0;
    matcher->comparisons_made = 0;
    matcher->cache_hits = 0;
    matcher->average_similarity = 0.0f;
    matcher->average_match_time_ns = 0.0f;
}