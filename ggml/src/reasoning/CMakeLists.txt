# Reasoning module for advanced cognitive capabilities

set(REASONING_SOURCES
    pln-core.c
    moses-core.c
    pattern-matcher.c
)

set(REASONING_HEADERS
    pln-core.h
    moses-core.h
    pattern-matcher.h
)

# Create reasoning library
add_library(ggml-reasoning SHARED ${REASONING_SOURCES})

target_include_directories(ggml-reasoning PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(ggml-reasoning
    ggml-base
    m
)

# Set public headers
set_target_properties(ggml-reasoning PROPERTIES
    PUBLIC_HEADER "${REASONING_HEADERS}"
)

# Install targets
install(TARGETS ggml-reasoning
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include/ggml/reasoning
)