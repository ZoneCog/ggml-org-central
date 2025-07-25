#pragma once

#include "ggml.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct cognitive_agent;
struct self_modification_engine;

// Emergent behavior types
enum behavior_type {
    BEHAVIOR_COORDINATION = 1,       // Spontaneous coordination between agents
    BEHAVIOR_SPECIALIZATION = 2,     // Role specialization emergence
    BEHAVIOR_COMMUNICATION = 3,      // New communication patterns
    BEHAVIOR_LEARNING = 4,           // Collective learning behaviors
    BEHAVIOR_ADAPTATION = 5,         // Environmental adaptation
    BEHAVIOR_HIERARCHY = 6,          // Hierarchical structure formation
    BEHAVIOR_SWARM = 7,             // Swarm intelligence patterns
    BEHAVIOR_CONSCIOUSNESS = 8       // Consciousness-like properties
};

// Consciousness assessment metrics
enum consciousness_metric {
    CONSCIOUSNESS_SELF_AWARENESS = 1,     // Ability to model self
    CONSCIOUSNESS_OTHER_AWARENESS = 2,    // Ability to model others
    CONSCIOUSNESS_META_COGNITION = 3,     // Thinking about thinking
    CONSCIOUSNESS_INTENTIONALITY = 4,     // Goal-directed behavior
    CONSCIOUSNESS_INTEGRATION = 5,        // Information integration
    CONSCIOUSNESS_ATTENTION = 6,          // Selective attention
    CONSCIOUSNESS_MEMORY = 7,             // Autobiographical memory
    CONSCIOUSNESS_REFLECTION = 8          // Self-reflection capabilities
};

// Behavior pattern structure
typedef struct behavior_pattern {
    enum behavior_type type;              // Type of behavior observed
    uint64_t pattern_id;                  // Unique pattern identifier
    uint64_t first_observed;              // When first detected
    uint64_t last_observed;               // When last observed
    float strength;                       // Strength of the pattern
    float frequency;                      // How often it occurs
    float stability;                      // How stable/consistent it is
    
    // Participating agents
    uint64_t* agent_ids;                  // IDs of participating agents
    size_t agent_count;                   // Number of participating agents
    
    // Pattern characteristics
    float* feature_vector;                // Pattern feature representation
    size_t feature_size;                  // Size of feature vector
    
    // Context information
    char description[256];                // Human-readable description
    float emergence_rate;                 // Rate of emergence
    bool is_persistent;                   // Whether pattern persists
} behavior_pattern;

// Network behavior analysis
typedef struct network_behavior_analyzer {
    struct ggml_context* ctx;
    
    // Pattern storage
    behavior_pattern* patterns;           // Detected patterns
    size_t pattern_count;                // Number of patterns
    size_t pattern_capacity;             // Pattern storage capacity
    
    // Detection parameters
    float detection_threshold;           // Minimum strength for detection
    uint32_t observation_window;        // Time window for analysis
    uint32_t min_agent_count;           // Minimum agents for pattern
    
    // Statistical tracking
    float pattern_diversity;            // Diversity of patterns observed
    float emergence_rate;               // Rate of new pattern emergence
    uint64_t total_observations;        // Total behavioral observations
    
    // Consciousness metrics
    float consciousness_scores[8];      // Scores for each consciousness metric
    float overall_consciousness;        // Overall consciousness assessment
    
    // Analysis state
    bool is_analyzing;                  // Whether analysis is active
    uint64_t analysis_start_time;      // When analysis started
    uint32_t analysis_cycles;          // Number of analysis cycles
} network_behavior_analyzer;

// Consciousness assessment system
typedef struct consciousness_assessor {
    struct ggml_context* ctx;
    
    // Assessment metrics
    float self_awareness_score;         // Self-awareness measurement
    float other_awareness_score;        // Other-awareness measurement
    float meta_cognition_score;         // Meta-cognitive abilities
    float intentionality_score;         // Goal-directed behavior
    float integration_score;            // Information integration
    float attention_score;              // Selective attention
    float memory_score;                 // Autobiographical memory
    float reflection_score;             // Self-reflection
    
    // Temporal tracking
    float* consciousness_history;       // Historical consciousness scores
    size_t history_size;               // Size of history buffer
    size_t history_capacity;           // Capacity of history buffer
    
    // Assessment parameters
    float assessment_threshold;         // Threshold for consciousness
    uint32_t assessment_interval;      // How often to assess
    float consciousness_trend;          // Trend in consciousness development
    
    // Validation metrics
    float assessment_confidence;        // Confidence in assessment
    uint32_t assessment_count;         // Number of assessments performed
    bool consciousness_detected;        // Whether consciousness is detected
    float overall_consciousness;        // Overall consciousness score
} consciousness_assessor;

// Emergent property measurement tools
typedef struct emergent_property_analyzer {
    struct ggml_context* ctx;
    
    // Property detection
    struct ggml_tensor* property_vectors; // Emergent property representations
    float* property_strengths;            // Strength of each property
    bool* property_active;               // Whether property is currently active
    size_t property_count;               // Number of properties tracked
    
    // Measurement parameters
    float emergence_threshold;           // Threshold for property emergence
    float persistence_threshold;        // Threshold for property persistence
    uint32_t stability_window;          // Window for stability assessment
    
    // Statistical analysis
    float property_complexity;          // Complexity of emergent properties
    float interaction_density;          // Density of agent interactions
    float information_flow;             // Information flow between agents
    
    // Network topology metrics
    float clustering_coefficient;       // Network clustering
    float path_length;                  // Average path length
    float centrality_variance;          // Variance in node centrality
    
    // Evolution tracking
    uint64_t* evolution_timestamps;     // When properties evolved
    float* evolution_rates;             // Rate of property evolution
    size_t evolution_count;             // Number of evolution events
} emergent_property_analyzer;

// Main behavior analysis engine
typedef struct behavior_analysis_engine {
    struct ggml_context* ctx;
    
    // Core components
    network_behavior_analyzer* network_analyzer;     // Network behavior analysis
    consciousness_assessor* consciousness_assessor;  // Consciousness assessment
    emergent_property_analyzer* property_analyzer;   // Emergent property analysis
    
    // Agent network tracking
    struct cognitive_agent** monitored_agents;      // Agents being monitored
    size_t monitored_count;                         // Number of monitored agents
    size_t monitoring_capacity;                     // Monitoring capacity
    
    // Analysis results
    float overall_emergence_score;       // Overall emergence assessment
    float network_complexity;           // Network complexity measure
    float collective_intelligence;       // Collective intelligence score
    
    // Performance metrics
    uint64_t behaviors_detected;        // Total behaviors detected
    uint64_t consciousness_assessments; // Total consciousness assessments
    uint64_t properties_analyzed;       // Total properties analyzed
    
    // Analysis parameters
    bool continuous_monitoring;         // Whether to monitor continuously
    uint32_t analysis_frequency;       // Frequency of analysis cycles
    float significance_threshold;       // Threshold for significant behavior
} behavior_analysis_engine;

// Core engine functions
behavior_analysis_engine* init_behavior_analysis_engine(struct ggml_context* ctx);
void cleanup_behavior_analysis_engine(behavior_analysis_engine* engine);

// Behavior pattern detection
bool detect_behavior_patterns(behavior_analysis_engine* engine);
behavior_pattern* analyze_agent_interactions(behavior_analysis_engine* engine,
                                            struct cognitive_agent** agents,
                                            size_t agent_count);
void classify_behavior_pattern(behavior_pattern* pattern);
void track_pattern_evolution(behavior_analysis_engine* engine,
                           behavior_pattern* pattern);

// Network-wide behavior analysis
void analyze_network_topology(behavior_analysis_engine* engine);
void measure_information_flow(behavior_analysis_engine* engine);
void assess_collective_intelligence(behavior_analysis_engine* engine);
float compute_network_complexity(behavior_analysis_engine* engine);

// Consciousness assessment
void assess_consciousness_metrics(behavior_analysis_engine* engine,
                                struct cognitive_agent* agent);
float measure_self_awareness(struct cognitive_agent* agent);
float measure_other_awareness(struct cognitive_agent* agent,
                             struct cognitive_agent** other_agents,
                             size_t other_count);
float measure_meta_cognition(struct cognitive_agent* agent);
float measure_intentionality(struct cognitive_agent* agent);
float compute_overall_consciousness(consciousness_assessor* assessor);

// Emergent property measurement
void detect_emergent_properties(behavior_analysis_engine* engine);
void measure_property_strength(emergent_property_analyzer* analyzer,
                              size_t property_index);
void track_property_evolution(emergent_property_analyzer* analyzer);
float assess_property_stability(emergent_property_analyzer* analyzer,
                               size_t property_index);

// Statistical analysis
void compute_behavior_statistics(behavior_analysis_engine* engine);
void analyze_emergence_patterns(behavior_analysis_engine* engine);
void generate_behavior_report(behavior_analysis_engine* engine);
void visualize_behavior_patterns(behavior_analysis_engine* engine);

// Predictive models
float predict_behavior_evolution(behavior_analysis_engine* engine,
                               behavior_pattern* pattern,
                               float time_horizon);
void model_consciousness_development(behavior_analysis_engine* engine);
void forecast_emergent_properties(behavior_analysis_engine* engine);

// Integration functions
bool add_agent_to_monitoring(behavior_analysis_engine* engine,
                           struct cognitive_agent* agent);
void remove_agent_from_monitoring(behavior_analysis_engine* engine,
                                uint64_t agent_id);
void process_behavior_analysis_cycle(behavior_analysis_engine* engine);

// Utility functions
void log_behavior_detection(behavior_analysis_engine* engine,
                          behavior_pattern* pattern);
void update_consciousness_history(consciousness_assessor* assessor,
                                float consciousness_score);
float compute_pattern_similarity(behavior_pattern* pattern1,
                                behavior_pattern* pattern2);

#ifdef __cplusplus
}
#endif