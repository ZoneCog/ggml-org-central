# Multi-modal module for Phase 4 advanced cognitive capabilities

set(MULTIMODAL_SOURCES
    multimodal-core.c
    cross-modal-reasoning.c
)

set(MULTIMODAL_HEADERS
    multimodal-core.h
    cross-modal-reasoning.h
)

# Create multimodal library
add_library(ggml-multimodal SHARED ${MULTIMODAL_SOURCES})

target_include_directories(ggml-multimodal PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/reasoning
)

target_link_libraries(ggml-multimodal
    ggml-base
    ggml-reasoning
    m
)

# Set public headers
set_target_properties(ggml-multimodal PROPERTIES
    PUBLIC_HEADER "${MULTIMODAL_HEADERS}"
)

# Install targets
install(TARGETS ggml-multimodal
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include/ggml/multimodal
)