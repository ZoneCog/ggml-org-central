#pragma once

#include "ggml.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations for integration
struct cognitive_agent;
struct hypergraph_memory;
struct advanced_grammar;
struct ggml_context;

// Multi-modal data types
enum modality_type {
    MODALITY_TEXT = 1,
    MODALITY_AUDIO = 2,
    MODALITY_VISUAL = 3,
    MODALITY_TACTILE = 4,
    MODALITY_COMPOSITE = 5
};

// Multi-modal tensor representation
typedef struct multimodal_tensor {
    struct ggml_tensor* tensor_data;       // Actual tensor data
    enum modality_type modality;    // Primary modality type
    uint32_t* modality_mask;       // Mask for multi-modal regions
    float* attention_weights;      // Cross-modal attention weights
    size_t sequence_length;        // Temporal sequence length
    uint32_t feature_dimensions;   // Feature dimensionality
    uint64_t timestamp;            // When tensor was created
    char* metadata;                // Additional metadata (JSON format)
} multimodal_tensor;

// Audio processing context (whisper.cpp integration)
typedef struct audio_processor {
    void* whisper_context;         // whisper.cpp context
    float* mel_spectrogram;        // Preprocessed audio features
    size_t mel_length;            // Length of mel spectrogram
    int32_t sample_rate;          // Audio sample rate
    float* audio_buffer;          // Raw audio buffer
    size_t buffer_size;           // Buffer size in samples
    bool is_initialized;          // Initialization status
} audio_processor;

// Visual processing context (ggml vision models)
typedef struct visual_processor {
    struct ggml_context* vision_ctx;      // ggml context for vision models
    struct ggml_tensor* image_embeddings; // Image feature embeddings
    uint32_t image_width;         // Image width
    uint32_t image_height;        // Image height
    uint32_t channels;            // Number of channels (RGB=3)
    float* pixel_data;            // Raw pixel data
    bool is_initialized;          // Initialization status
} visual_processor;

// Cross-modal attention mechanism
typedef struct cross_modal_attention {
    struct ggml_tensor* query_projection;  // Query projection matrix
    struct ggml_tensor* key_projection;    // Key projection matrix
    struct ggml_tensor* value_projection;  // Value projection matrix
    struct ggml_tensor* output_projection; // Output projection matrix
    uint32_t attention_heads;       // Number of attention heads
    uint32_t head_dimension;        // Dimension per attention head
    float attention_dropout;        // Dropout rate for attention
    float temperature;              // Attention temperature scaling
} cross_modal_attention;

// Multi-modal fusion strategy
enum fusion_strategy {
    FUSION_EARLY = 1,              // Early fusion at feature level
    FUSION_LATE = 2,               // Late fusion at decision level
    FUSION_INTERMEDIATE = 3,       // Intermediate fusion at multiple levels
    FUSION_ADAPTIVE = 4            // Adaptive fusion based on context
};

// Multi-modal processor core
typedef struct multimodal_processor {
    audio_processor* audio;         // Audio processing component
    visual_processor* visual;       // Visual processing component
    cross_modal_attention* attention; // Cross-modal attention
    struct ggml_context* fusion_ctx;      // Context for fusion operations
    enum fusion_strategy strategy;  // Fusion strategy
    multimodal_tensor* unified_representation; // Unified multi-modal tensor
    uint32_t max_sequence_length;  // Maximum sequence length
    bool is_active;                // Processor status
} multimodal_processor;

// Multi-modal memory integration
typedef struct multimodal_memory {
    struct hypergraph_memory* base_memory; // Base hypergraph memory
    multimodal_tensor** modal_tensors;     // Stored multi-modal tensors
    size_t tensor_count;                   // Number of stored tensors
    size_t tensor_capacity;                // Storage capacity
    float* similarity_cache;               // Cross-modal similarity cache
    uint32_t cache_size;                   // Cache size
} multimodal_memory;

// Function declarations

// Core multi-modal processor functions
multimodal_processor* create_multimodal_processor(uint32_t max_sequence_length, enum fusion_strategy strategy);
void destroy_multimodal_processor(multimodal_processor* processor);
bool initialize_multimodal_components(multimodal_processor* processor);
bool process_multimodal_input(multimodal_processor* processor, void* input_data, enum modality_type modality, multimodal_tensor** output);

// Audio processing with whisper.cpp integration
bool initialize_audio_processor(audio_processor* audio, int sample_rate);
bool process_audio_input(audio_processor* audio, float* audio_data, size_t data_length, struct ggml_tensor** audio_features);
bool extract_mel_spectrogram(audio_processor* audio, float* audio_data, size_t length);
bool transcribe_audio_to_text(audio_processor* audio, char** transcription, float* confidence);
void cleanup_audio_processor(audio_processor* audio);

// Visual processing with ggml vision models
bool initialize_visual_processor(visual_processor* visual, uint32_t width, uint32_t height, uint32_t channels);
bool process_image_input(visual_processor* visual, float* image_data, struct ggml_tensor** image_features);
bool extract_visual_features(visual_processor* visual, float* pixel_data);
bool detect_visual_objects(visual_processor* visual, void** detection_results, size_t* detection_count);
void cleanup_visual_processor(visual_processor* visual);

// Unified multi-modal tensor operations
multimodal_tensor* create_multimodal_tensor(enum modality_type modality, uint32_t feature_dims, size_t sequence_length);
void destroy_multimodal_tensor(multimodal_tensor* tensor);
bool combine_modal_tensors(multimodal_tensor** input_tensors, size_t tensor_count, enum fusion_strategy strategy, multimodal_tensor** output);
bool align_temporal_sequences(multimodal_tensor** tensors, size_t count, size_t target_length);

// Cross-modal attention mechanisms
cross_modal_attention* create_cross_modal_attention(uint32_t model_dim, uint32_t num_heads, float dropout);
void destroy_cross_modal_attention(cross_modal_attention* attention);
bool compute_cross_modal_attention(cross_modal_attention* attention, multimodal_tensor* query, multimodal_tensor* key, multimodal_tensor* value, struct ggml_tensor** output);
bool apply_attention_to_modalities(cross_modal_attention* attention, multimodal_tensor** modalities, size_t modality_count, struct ggml_tensor** attended_output);

// Multi-modal fusion strategies
bool early_fusion(multimodal_tensor** input_tensors, size_t count, struct ggml_context* ctx, struct ggml_tensor** fused_output);
bool late_fusion(struct ggml_tensor** modal_decisions, size_t count, float* fusion_weights, struct ggml_tensor** final_decision);
bool intermediate_fusion(multimodal_tensor** input_tensors, size_t count, uint32_t* fusion_layers, size_t layer_count, struct ggml_tensor** output);
bool adaptive_fusion(multimodal_tensor** input_tensors, size_t count, float* context_weights, struct ggml_tensor** output);

// Multi-modal memory integration
multimodal_memory* create_multimodal_memory(struct hypergraph_memory* base_memory, size_t capacity);
void destroy_multimodal_memory(multimodal_memory* memory);
bool store_multimodal_tensor(multimodal_memory* memory, multimodal_tensor* tensor, const char* identifier);
bool retrieve_multimodal_tensor(multimodal_memory* memory, const char* identifier, multimodal_tensor** tensor);
bool search_multimodal_memory(multimodal_memory* memory, multimodal_tensor* query, float threshold, multimodal_tensor*** results, size_t* result_count);

// Cross-modal similarity and matching
float compute_cross_modal_similarity(multimodal_tensor* tensor1, multimodal_tensor* tensor2);
bool align_cross_modal_features(multimodal_tensor* tensor1, multimodal_tensor* tensor2, struct ggml_tensor** aligned1, struct ggml_tensor** aligned2);
bool detect_cross_modal_correspondences(multimodal_tensor** tensors, size_t count, float** correspondence_matrix);

// Multi-modal cognitive grammar extensions
bool extend_grammar_with_modalities(struct advanced_grammar* grammar, enum modality_type* supported_modalities, size_t modality_count);
bool parse_multimodal_expression(struct advanced_grammar* grammar, multimodal_tensor* input, char** parsed_output, float* confidence);
bool generate_multimodal_response(struct advanced_grammar* grammar, const char* intent, enum modality_type target_modality, multimodal_tensor** response);

// Integration with cognitive agents
bool install_multimodal_processor(struct cognitive_agent* agent, multimodal_processor* processor);
bool process_multimodal_communication(struct cognitive_agent* sender, struct cognitive_agent* receiver, multimodal_tensor* message);
bool enable_multimodal_reasoning(struct cognitive_agent* agent, bool enable_cross_modal);

// Performance optimization
bool optimize_multimodal_memory_usage(multimodal_processor* processor);
bool cache_frequent_modal_patterns(multimodal_processor* processor, multimodal_tensor** patterns, size_t pattern_count);
bool batch_multimodal_processing(multimodal_processor* processor, void** inputs, enum modality_type* modalities, size_t batch_size, multimodal_tensor*** outputs);

// Serialization and sharing
bool serialize_multimodal_tensor(multimodal_tensor* tensor, uint8_t** serialized_data, size_t* data_size);
multimodal_tensor* deserialize_multimodal_tensor(uint8_t* serialized_data, size_t data_size);
bool share_multimodal_knowledge(struct cognitive_agent** agents, size_t agent_count, multimodal_tensor** knowledge, size_t knowledge_count);

#ifdef __cplusplus
}
#endif