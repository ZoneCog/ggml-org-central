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

// Hierarchical agent organization levels
enum agent_hierarchy_level {
    LEVEL_WORKER = 0,      // Base level workers performing tasks
    LEVEL_SUPERVISOR = 1,  // Supervisors coordinating workers
    LEVEL_MANAGER = 2,     // Managers overseeing supervisors  
    LEVEL_DIRECTOR = 3,    // Directors managing system regions
    LEVEL_EXECUTIVE = 4    // Executive level system coordination
};

// Agent specialization types for large-scale networks
enum agent_specialization {
    SPEC_GENERAL = 0,           // General purpose agents
    SPEC_MEMORY = 1,            // Memory management specialists
    SPEC_COMMUNICATION = 2,     // Communication routing specialists
    SPEC_REASONING = 3,         // Reasoning and inference specialists
    SPEC_ATTENTION = 4,         // Attention allocation specialists
    SPEC_COORDINATION = 5,      // Inter-agent coordination specialists
    SPEC_MONITORING = 6,        // System monitoring and diagnostics
    SPEC_LOAD_BALANCING = 7     // Load balancing and optimization
};

// Hierarchical organization structure
typedef struct hierarchical_organization {
    uint64_t org_id;                    // Organization identifier
    char name[128];                     // Organization name
    enum agent_hierarchy_level max_level; // Maximum hierarchy level
    
    // Agent distribution across levels
    uint32_t agents_per_level[5];       // Count of agents at each level
    uint64_t* level_agents[5];          // Agent IDs at each level
    uint32_t level_capacity[5];         // Capacity at each level
    
    // Specialization tracking
    uint32_t specialists_count[8];      // Count by specialization type
    uint64_t* specialists[8];           // Specialist agent IDs by type
    uint32_t specialist_capacity[8];    // Capacity for each specialization
    
    // Performance metrics
    float coordination_efficiency;       // How well the hierarchy coordinates
    float load_distribution;            // How evenly load is distributed
    float communication_latency;        // Average communication latency
    float throughput;                   // Messages processed per second
    
    // Load balancing state
    float level_load[5];                // Current load at each level
    float specialist_load[8];           // Current load for each specialization
    uint64_t last_rebalance_time;       // Last load rebalancing timestamp
    
    struct ggml_context* ctx;           // GGML context for tensor operations
} hierarchical_organization;

// Coordination protocol for large networks
typedef struct coordination_protocol {
    uint64_t protocol_id;               // Protocol identifier
    char name[64];                      // Protocol name
    enum agent_hierarchy_level scope;   // Scope of coordination
    
    // Message routing
    float routing_weights[5][5];        // Routing weights between levels
    uint32_t max_hops;                  // Maximum message hops allowed
    float timeout_ms;                   // Message timeout
    
    // Load balancing parameters
    float load_threshold;               // Threshold for load rebalancing
    uint32_t rebalance_interval_ms;     // Rebalancing check interval
    float migration_cost;               // Cost of moving agents between levels
    
    // Performance monitoring
    uint64_t messages_routed;           // Total messages routed
    uint64_t failed_routes;             // Failed routing attempts
    float average_latency;              // Average message latency
    
    struct ggml_context* ctx;           // GGML context
} coordination_protocol;

// Performance profiling for large-scale networks
typedef struct performance_profiler {
    uint64_t profiler_id;               // Profiler identifier
    bool active;                        // Whether profiling is active
    
    // Network scale metrics
    uint32_t total_agents;              // Total agents in network
    uint32_t active_agents;             // Currently active agents
    uint32_t max_concurrent_agents;     // Peak concurrent agents
    
    // Performance measurements
    float messages_per_second;          // Message throughput
    float cpu_utilization;              // Average CPU usage
    float memory_utilization;           // Memory usage percentage
    float network_bandwidth;            // Network bandwidth usage
    
    // Scalability metrics
    float linear_scaling_factor;        // How close to linear scaling
    float response_time_p99;            // 99th percentile response time
    float error_rate;                   // Error rate percentage
    
    // Performance history
    float* throughput_history;          // Historical throughput data
    float* latency_history;             // Historical latency data
    uint32_t history_size;              // Size of history buffers
    uint32_t history_index;             // Current index in history
    
    struct ggml_context* ctx;           // GGML context
} performance_profiler;

// Function declarations

// Hierarchical organization management
hierarchical_organization* create_hierarchical_organization(
    const char* name, 
    enum agent_hierarchy_level max_level,
    uint32_t initial_capacity
);

bool add_agent_to_hierarchy(
    hierarchical_organization* org,
    uint64_t agent_id,
    enum agent_hierarchy_level level,
    enum agent_specialization specialization
);

bool remove_agent_from_hierarchy(
    hierarchical_organization* org,
    uint64_t agent_id
);

bool promote_agent(
    hierarchical_organization* org,
    uint64_t agent_id,
    enum agent_hierarchy_level new_level
);

bool specialize_agent(
    hierarchical_organization* org,
    uint64_t agent_id,
    enum agent_specialization specialization
);

// Load balancing across hierarchical levels
bool rebalance_hierarchy_load(hierarchical_organization* org);
float calculate_load_distribution_efficiency(hierarchical_organization* org);
bool migrate_agent_between_levels(
    hierarchical_organization* org,
    uint64_t agent_id,
    enum agent_hierarchy_level target_level
);

// Coordination protocols for large networks
coordination_protocol* create_coordination_protocol(
    const char* name,
    enum agent_hierarchy_level scope
);

bool route_message_through_hierarchy(
    coordination_protocol* protocol,
    hierarchical_organization* org,
    uint64_t from_agent,
    uint64_t to_agent,
    const void* message_data,
    size_t message_size
);

bool optimize_routing_weights(coordination_protocol* protocol);
float measure_coordination_efficiency(
    coordination_protocol* protocol,
    hierarchical_organization* org
);

// Performance profiling and optimization
performance_profiler* create_performance_profiler(void);
bool start_profiling(performance_profiler* profiler);
bool stop_profiling(performance_profiler* profiler);

bool update_hierarchy_performance_metrics(
    performance_profiler* profiler,
    hierarchical_organization* org
);

float calculate_scaling_efficiency(
    performance_profiler* profiler,
    uint32_t baseline_agents,
    uint32_t current_agents
);

bool generate_performance_report(
    performance_profiler* profiler,
    const char* output_file
);

// Large-scale optimization
bool optimize_for_scale(
    hierarchical_organization* org,
    coordination_protocol* protocol,
    uint32_t target_agent_count
);

bool enable_parallel_processing(hierarchical_organization* org);
bool optimize_memory_management(hierarchical_organization* org);
bool optimize_tensor_operations(hierarchical_organization* org);

// Cleanup functions
void destroy_hierarchical_organization(hierarchical_organization* org);
void destroy_coordination_protocol(coordination_protocol* protocol);
void destroy_performance_profiler(performance_profiler* profiler);

// Performance target validation for 1000+ agents
bool validate_scalability_targets(
    performance_profiler* profiler,
    uint32_t target_agents,
    float target_throughput,
    float max_latency_ms
);

#ifdef __cplusplus
}
#endif