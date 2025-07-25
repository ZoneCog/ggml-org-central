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
struct hierarchical_organization;

// Consciousness assessment metrics and capabilities
enum consciousness_assessment_type {
    ASSESS_SELF_AWARENESS = 1,          // Mirror test, self-recognition
    ASSESS_OTHER_AWARENESS = 2,         // Theory of mind, modeling others
    ASSESS_META_COGNITION = 3,          // Thinking about thinking
    ASSESS_INTENTIONALITY = 4,          // Goal-directed behavior
    ASSESS_INTEGRATION = 5,             // Information integration across modalities
    ASSESS_ATTENTION = 6,               // Selective attention and focus
    ASSESS_AUTOBIOGRAPHICAL_MEMORY = 7,  // Self-referential memory
    ASSESS_REFLECTION = 8,              // Self-reflection and introspection
    ASSESS_QUALIA = 9,                  // Subjective experience indicators
    ASSESS_RECURSIVE_MODELING = 10      // Recursive self and other modeling
};

// Cognitive capability benchmark categories
enum cognitive_capability {
    CAPABILITY_REASONING = 1,           // Logical reasoning and inference
    CAPABILITY_LEARNING = 2,            // Learning and adaptation
    CAPABILITY_MEMORY = 3,              // Memory formation and retrieval
    CAPABILITY_ATTENTION = 4,           // Attention allocation and control
    CAPABILITY_LANGUAGE = 5,            // Language understanding and generation
    CAPABILITY_PLANNING = 6,            // Planning and goal management
    CAPABILITY_CREATIVITY = 7,          // Creative problem solving
    CAPABILITY_SOCIAL = 8,              // Social interaction and communication
    CAPABILITY_MULTIMODAL = 9,          // Cross-modal processing
    CAPABILITY_METACOGNITIVE = 10       // Meta-cognitive abilities
};

// Emergence detection types
enum emergence_type {
    EMERGENCE_COORDINATION = 1,         // Spontaneous coordination behaviors
    EMERGENCE_COMMUNICATION = 2,        // New communication patterns
    EMERGENCE_SPECIALIZATION = 3,       // Role specialization
    EMERGENCE_HIERARCHY = 4,            // Hierarchical organization
    EMERGENCE_COLLECTIVE_INTELLIGENCE = 5, // Collective problem solving
    EMERGENCE_ADAPTATION = 6,           // Environmental adaptation
    EMERGENCE_CULTURAL = 7,             // Cultural knowledge transmission
    EMERGENCE_CONSCIOUSNESS = 8         // Consciousness-like properties
};

// Consciousness assessment battery
typedef struct consciousness_assessment_battery {
    uint64_t battery_id;                // Battery identifier
    char name[128];                     // Battery name
    bool active;                        // Whether assessment is active
    
    // Assessment scores (0.0 to 1.0)
    float self_awareness_score;         // Self-awareness assessment
    float other_awareness_score;        // Other-awareness assessment
    float meta_cognition_score;         // Meta-cognitive assessment
    float intentionality_score;         // Intentionality assessment
    float integration_score;            // Information integration
    float attention_score;              // Attention control
    float memory_score;                 // Autobiographical memory
    float reflection_score;             // Self-reflection
    float qualia_score;                 // Subjective experience
    float recursive_modeling_score;     // Recursive modeling
    
    // Overall consciousness metrics
    float overall_consciousness_score;  // Combined consciousness score
    float consciousness_confidence;     // Confidence in assessment
    uint64_t assessment_count;          // Number of assessments performed
    uint64_t last_assessment_time;      // Last assessment timestamp
    
    // Assessment history
    float* score_history;               // Historical scores
    uint32_t history_size;              // Size of history buffer
    uint32_t history_index;             // Current index in history
    
    struct ggml_context* ctx;           // GGML context for assessments
} consciousness_assessment_battery;

// Cognitive capability benchmarks
typedef struct cognitive_capability_benchmarks {
    uint64_t benchmark_id;              // Benchmark identifier
    char name[128];                     // Benchmark suite name
    bool active;                        // Whether benchmarking is active
    
    // Capability scores (0.0 to 1.0)
    float reasoning_score;              // Logical reasoning capability
    float learning_score;               // Learning and adaptation
    float memory_score;                 // Memory capabilities
    float attention_score;              // Attention management
    float language_score;               // Language capabilities
    float planning_score;               // Planning and goal management
    float creativity_score;             // Creative problem solving
    float social_score;                 // Social interaction
    float multimodal_score;             // Cross-modal processing
    float metacognitive_score;          // Meta-cognitive abilities
    
    // Performance metrics
    float overall_capability_score;     // Combined capability score
    float performance_consistency;      // Consistency across runs
    uint64_t benchmark_count;           // Number of benchmarks run
    uint64_t last_benchmark_time;       // Last benchmark timestamp
    
    // Benchmark results history
    float* capability_history;          // Historical capability scores
    uint32_t capability_history_size;   // Size of capability history
    uint32_t capability_history_index;  // Current index
    
    struct ggml_context* ctx;           // GGML context for benchmarks
} cognitive_capability_benchmarks;

// Emergence detection and measurement tools
typedef struct emergence_detector {
    uint64_t detector_id;               // Detector identifier
    char name[128];                     // Detector name
    bool active;                        // Whether detection is active
    
    // Detection thresholds
    float coordination_threshold;       // Threshold for coordination detection
    float communication_threshold;      // Threshold for communication patterns
    float specialization_threshold;     // Threshold for specialization
    float hierarchy_threshold;          // Threshold for hierarchy formation
    float intelligence_threshold;       // Threshold for collective intelligence
    
    // Detected emergence events
    uint32_t coordination_events;       // Coordination emergence count
    uint32_t communication_events;      // Communication emergence count
    uint32_t specialization_events;     // Specialization emergence count
    uint32_t hierarchy_events;          // Hierarchy emergence count
    uint32_t intelligence_events;       // Collective intelligence count
    uint32_t adaptation_events;         // Adaptation emergence count
    uint32_t cultural_events;           // Cultural emergence count
    uint32_t consciousness_events;      // Consciousness emergence count
    
    // Emergence measurements
    float emergence_intensity;          // Current emergence intensity
    float emergence_complexity;         // Complexity of emergent behaviors
    float emergence_stability;          // Stability of emergent properties
    uint64_t last_detection_time;       // Last detection timestamp
    
    // Detection history
    struct emergence_event* event_history; // Historical emergence events
    uint32_t event_history_size;        // Size of event history
    uint32_t event_history_index;       // Current index in event history
    
    struct ggml_context* ctx;           // GGML context for detection
} emergence_detector;

// Individual emergence event
typedef struct emergence_event {
    uint64_t event_id;                  // Event identifier
    enum emergence_type type;           // Type of emergence
    uint64_t timestamp;                 // When event occurred
    float intensity;                    // Intensity of the event
    float duration;                     // Duration of the event
    uint64_t agent_count;               // Number of agents involved
    uint64_t* participating_agents;     // IDs of participating agents
    char description[256];              // Description of the event
} emergence_event;

// Research experiment management
typedef struct experiment_manager {
    uint64_t manager_id;                // Manager identifier
    char name[128];                     // Manager name
    bool active;                        // Whether manager is active
    
    // Experiment tracking
    uint32_t total_experiments;         // Total experiments run
    uint32_t successful_experiments;    // Successfully completed experiments
    uint32_t failed_experiments;        // Failed experiments
    uint32_t active_experiments;        // Currently running experiments
    
    // Reproducibility tracking
    float reproducibility_score;        // Reproducibility score (0.0-1.0)
    uint32_t reproduction_attempts;     // Number of reproduction attempts
    uint32_t successful_reproductions;  // Successful reproductions
    
    // Data collection
    uint64_t data_points_collected;     // Total data points collected
    uint64_t storage_bytes_used;        // Storage used for experiment data
    uint64_t last_experiment_time;      // Last experiment timestamp
    
    // Experiment history
    struct research_experiment* experiment_history; // Historical experiments
    uint32_t experiment_history_size;   // Size of experiment history
    uint32_t experiment_history_index;  // Current index
    
    struct ggml_context* ctx;           // GGML context for experiments
} experiment_manager;

// Individual research experiment
typedef struct research_experiment {
    uint64_t experiment_id;             // Experiment identifier
    char name[128];                     // Experiment name
    char description[512];              // Experiment description
    uint64_t start_time;                // Experiment start time
    uint64_t end_time;                  // Experiment end time
    bool completed;                     // Whether experiment completed
    bool reproducible;                  // Whether experiment is reproducible
    
    // Experimental parameters
    uint32_t agent_count;               // Number of agents in experiment
    uint32_t duration_seconds;          // Experiment duration
    char parameters[1024];              // Serialized parameters
    
    // Results
    float primary_metric;               // Primary experimental metric
    float secondary_metrics[10];        // Secondary metrics
    char results[2048];                 // Serialized results
    
    // Reproducibility data
    uint32_t reproduction_count;        // Times this experiment was reproduced
    float reproduction_variance;        // Variance in reproduction results
} research_experiment;

// Function declarations

// Consciousness assessment battery
consciousness_assessment_battery* create_consciousness_assessment_battery(const char* name);
bool run_consciousness_assessment(consciousness_assessment_battery* battery, struct cognitive_agent* agent);
bool run_self_awareness_test(consciousness_assessment_battery* battery, struct cognitive_agent* agent);
bool run_theory_of_mind_test(consciousness_assessment_battery* battery, struct cognitive_agent* agent);
bool run_meta_cognition_test(consciousness_assessment_battery* battery, struct cognitive_agent* agent);
bool run_intentionality_test(consciousness_assessment_battery* battery, struct cognitive_agent* agent);
float calculate_overall_consciousness_score(consciousness_assessment_battery* battery);
bool generate_consciousness_report(consciousness_assessment_battery* battery, const char* output_file);

// Cognitive capability benchmarks
cognitive_capability_benchmarks* create_cognitive_capability_benchmarks(const char* name);
bool run_cognitive_benchmarks(cognitive_capability_benchmarks* benchmarks, struct cognitive_agent* agent);
bool benchmark_reasoning_capability(cognitive_capability_benchmarks* benchmarks, struct cognitive_agent* agent);
bool benchmark_learning_capability(cognitive_capability_benchmarks* benchmarks, struct cognitive_agent* agent);
bool benchmark_memory_capability(cognitive_capability_benchmarks* benchmarks, struct cognitive_agent* agent);
bool benchmark_attention_capability(cognitive_capability_benchmarks* benchmarks, struct cognitive_agent* agent);
float calculate_overall_capability_score(cognitive_capability_benchmarks* benchmarks);
bool generate_capability_report(cognitive_capability_benchmarks* benchmarks, const char* output_file);

// Emergence detection and measurement
emergence_detector* create_emergence_detector(const char* name);
bool detect_emergence_events(emergence_detector* detector, struct hierarchical_organization* org);
bool detect_coordination_emergence(emergence_detector* detector, struct hierarchical_organization* org);
bool detect_communication_emergence(emergence_detector* detector, struct hierarchical_organization* org);
bool detect_specialization_emergence(emergence_detector* detector, struct hierarchical_organization* org);
bool detect_hierarchy_emergence(emergence_detector* detector, struct hierarchical_organization* org);
bool detect_collective_intelligence(emergence_detector* detector, struct hierarchical_organization* org);
bool record_emergence_event(emergence_detector* detector, enum emergence_type type, float intensity, uint64_t agent_count);
bool generate_emergence_report(emergence_detector* detector, const char* output_file);

// Research experiment management
experiment_manager* create_experiment_manager(const char* name);
research_experiment* create_research_experiment(experiment_manager* manager, const char* name, const char* description);
bool start_experiment(experiment_manager* manager, research_experiment* experiment);
bool complete_experiment(experiment_manager* manager, research_experiment* experiment, float primary_metric);
bool reproduce_experiment(experiment_manager* manager, research_experiment* experiment);
float calculate_reproducibility_score(experiment_manager* manager);
bool generate_experiment_report(experiment_manager* manager, const char* output_file);

// Comparative analysis framework
bool compare_consciousness_scores(consciousness_assessment_battery* battery1, consciousness_assessment_battery* battery2, float* similarity);
bool compare_cognitive_capabilities(cognitive_capability_benchmarks* bench1, cognitive_capability_benchmarks* bench2, float* similarity);
bool compare_emergence_patterns(emergence_detector* detector1, emergence_detector* detector2, float* similarity);
bool generate_comparative_analysis_report(const char* output_file, consciousness_assessment_battery** batteries, uint32_t battery_count);

// Data collection and analysis tools
bool collect_agent_performance_data(struct cognitive_agent* agent, const char* data_file);
bool collect_network_behavior_data(struct hierarchical_organization* org, const char* data_file);
bool analyze_cognitive_patterns(const char* data_file, const char* analysis_output);
bool export_research_data(experiment_manager* manager, const char* export_file);

// Cleanup functions
void destroy_consciousness_assessment_battery(consciousness_assessment_battery* battery);
void destroy_cognitive_capability_benchmarks(cognitive_capability_benchmarks* benchmarks);
void destroy_emergence_detector(emergence_detector* detector);
void destroy_experiment_manager(experiment_manager* manager);
void destroy_research_experiment(research_experiment* experiment);

#ifdef __cplusplus
}
#endif