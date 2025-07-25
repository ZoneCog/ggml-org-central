cmake_minimum_required(VERSION 3.14)

# Build autonomy library - including Phase 5 components
add_library(ggml-autonomy SHARED
    self-modification.c
    behavior-analysis.c
    hierarchical-organization.c
    research-platform.c
)

target_include_directories(ggml-autonomy PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/reasoning
    ${CMAKE_SOURCE_DIR}/examples/cognitive-agents
)

target_link_libraries(ggml-autonomy
    ggml-base
    ggml-reasoning
    ${CMAKE_THREAD_LIBS_INIT}
    m  # Math library for floating point operations
)

# Set library properties
set_target_properties(ggml-autonomy PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
    PUBLIC_HEADER "self-modification.h;behavior-analysis.h;hierarchical-organization.h;research-platform.h"
)

# Add compile definitions
target_compile_definitions(ggml-autonomy PRIVATE
    _GNU_SOURCE
)

# Installation rules
install(TARGETS ggml-autonomy
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include/ggml
)