#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define RPC_PROTO_MAJOR_VERSION    2
#define RPC_PROTO_MINOR_VERSION    1
#define RPC_PROTO_PATCH_VERSION    0
#define GGML_RPC_MAX_SERVERS       16

// Cognitive RPC extensions
#define GGML_RPC_MAX_AGENTS        64
#define GGML_RPC_COGNITIVE_META_SIZE 256

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_rpc_init(const char * endpoint);
GGML_BACKEND_API bool ggml_backend_is_rpc(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint);

GGML_BACKEND_API void ggml_backend_rpc_get_device_memory(const char * endpoint, size_t * free, size_t * total);

GGML_BACKEND_API void ggml_backend_rpc_start_server(ggml_backend_t backend, const char * endpoint,
                                                    const char * cache_dir,
                                                    size_t free_mem, size_t total_mem);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_reg(void);

GGML_BACKEND_API ggml_backend_dev_t ggml_backend_rpc_add_device(const char * endpoint);

// Cognitive RPC extensions
typedef struct ggml_rpc_cognitive_meta {
    float attention_weight;          // Economic attention value
    uint32_t cognitive_type;         // Type of cognitive operation
    uint64_t source_agent_id;        // Originating agent
    uint64_t target_agent_id;        // Target agent
    char meta_context[256];          // Context information
    float salience_score;            // Relevance measure
    uint32_t recursion_depth;        // Self-reference depth
    uint64_t timestamp;              // When created
    float cognitive_capacity;        // Agent's processing capacity
    uint32_t network_hop_count;      // Number of network hops
} ggml_rpc_cognitive_meta;

// Cognitive RPC agent discovery
typedef struct ggml_rpc_agent_info {
    uint64_t agent_id;
    char endpoint[256];
    float cognitive_capacity;
    float current_load;
    uint32_t supported_types;        // Bitmask of supported cognitive types
    uint64_t last_heartbeat;
    float network_latency;
} ggml_rpc_agent_info;

// Cognitive network topology
typedef struct ggml_rpc_network_topology {
    ggml_rpc_agent_info agents[GGML_RPC_MAX_AGENTS];
    size_t agent_count;
    float network_performance_matrix[GGML_RPC_MAX_AGENTS][GGML_RPC_MAX_AGENTS];
    uint64_t last_update;
} ggml_rpc_network_topology;

// Cognitive RPC API
GGML_BACKEND_API int ggml_backend_rpc_send_cognitive_tensor(const char * endpoint,
                                                           struct ggml_tensor * tensor,
                                                           const ggml_rpc_cognitive_meta * meta);

GGML_BACKEND_API int ggml_backend_rpc_register_agent(const char * endpoint,
                                                     uint64_t agent_id,
                                                     float cognitive_capacity,
                                                     uint32_t supported_types);

GGML_BACKEND_API int ggml_backend_rpc_discover_agents(const char * endpoint,
                                                     ggml_rpc_agent_info * agents,
                                                     size_t * agent_count);

GGML_BACKEND_API int ggml_backend_rpc_update_network_topology(ggml_rpc_network_topology * topology);

GGML_BACKEND_API int ggml_backend_rpc_route_cognitive_message(const ggml_rpc_network_topology * topology,
                                                             uint64_t source_agent,
                                                             uint64_t target_agent,
                                                             uint32_t cognitive_type,
                                                             char * best_route,
                                                             size_t route_size);

GGML_BACKEND_API float ggml_backend_rpc_get_network_latency(const char * endpoint);

GGML_BACKEND_API int ggml_backend_rpc_optimize_bandwidth(const char * endpoint, bool enable_compression);

GGML_BACKEND_API void ggml_backend_rpc_start_performance_monitor(const char * endpoint);

#ifdef  __cplusplus
}
#endif
