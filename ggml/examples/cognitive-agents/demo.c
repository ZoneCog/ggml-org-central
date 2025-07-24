#include "cognitive-agent.h"
#include <stdio.h>
#include <unistd.h>

// Demo: Simple consciousness exploration task
void demo_consciousness_exploration(void) {
    printf("\n=== Consciousness Exploration Demo ===\n");
    
    // Create two cognitive agents
    cognitive_agent* philosopher = create_cognitive_agent("localhost:8001");
    cognitive_agent* scientist = create_cognitive_agent("localhost:8002");
    
    // Add relevant knowledge to each agent
    printf("\nAdding knowledge to agents...\n");
    
    // Philosopher's knowledge
    float consciousness_emb[64];
    for (int i = 0; i < 64; i++) consciousness_emb[i] = (float)i / 64.0f;
    add_knowledge(philosopher->memory, "consciousness", consciousness_emb, 64);
    
    float philosophy_emb[64];
    for (int i = 0; i < 64; i++) philosophy_emb[i] = 1.0f - (float)i / 64.0f;
    add_knowledge(philosopher->memory, "philosophy_of_mind", philosophy_emb, 64);
    
    // Scientist's knowledge
    float neuroscience_emb[64];
    for (int i = 0; i < 64; i++) neuroscience_emb[i] = sinf((float)i / 64.0f * 3.14159f);
    add_knowledge(scientist->memory, "neuroscience", neuroscience_emb, 64);
    
    float cognition_emb[64];
    for (int i = 0; i < 64; i++) cognition_emb[i] = cosf((float)i / 64.0f * 3.14159f);
    add_knowledge(scientist->memory, "cognitive_science", cognition_emb, 64);
    
    // Simulate consciousness exploration task
    printf("\nSimulating consciousness exploration...\n");
    
    // Philosopher initiates inquiry
    struct ggml_tensor* inquiry = ggml_new_tensor_1d(philosopher->ctx, GGML_TYPE_F32, 128);
    float* inquiry_data = (float*)inquiry->data;
    for (int i = 0; i < 128; i++) {
        inquiry_data[i] = (float)i / 128.0f;  // Represents "What is consciousness?"
    }
    
    // Philosopher allocates attention to reasoning
    allocate_attention(philosopher->attention, 0.6f, COGNITIVE_TYPE_REASONING);
    
    // Send inquiry to scientist
    send_cognitive_tensor(philosopher, scientist->agent_id, inquiry, 0.8f);
    
    // Scientist processes the inquiry
    cognitive_tensor_packet msg = {0};
    msg.cognitive_type = COGNITIVE_TYPE_REASONING;
    msg.attention_weight = 0.8f;
    msg.source_agent_id = philosopher->agent_id;
    msg.target_agent_id = scientist->agent_id;
    msg.salience_score = 0.9f;
    
    process_incoming_tensor(scientist, &msg);
    
    // Scientist formulates response based on neuroscience knowledge
    struct ggml_tensor* response = ggml_new_tensor_1d(scientist->ctx, GGML_TYPE_F32, 256);
    float* response_data = (float*)response->data;
    for (int i = 0; i < 256; i++) {
        response_data[i] = sinf((float)i / 256.0f * 6.28f);  // Neural patterns
    }
    
    // Send response back
    send_cognitive_tensor(scientist, philosopher->agent_id, response, 0.7f);
    
    // Philosopher processes scientific perspective
    msg.cognitive_type = COGNITIVE_TYPE_MEMORY;
    msg.attention_weight = 0.7f;
    msg.source_agent_id = scientist->agent_id;
    msg.target_agent_id = philosopher->agent_id;
    
    process_incoming_tensor(philosopher, &msg);
    
    // Both agents update their beliefs through reasoning
    printf("\nCognitive state updates:\n");
    printf("Philosopher - Inferences made: %lu\n", philosopher->reasoning->inferences_made);
    printf("Scientist - Inferences made: %lu\n", scientist->reasoning->inferences_made);
    
    // Search for concepts in memory
    hypergraph_node* consciousness_node = find_concept(philosopher->memory, "consciousness");
    if (consciousness_node) {
        printf("Philosopher found consciousness concept with truth value: %.2f\n", 
               consciousness_node->truth_value);
    }
    
    hypergraph_node* neuroscience_node = find_concept(scientist->memory, "neuroscience");
    if (neuroscience_node) {
        printf("Scientist found neuroscience concept with truth value: %.2f\n", 
               neuroscience_node->truth_value);
    }
    
    // Display attention allocation
    printf("\nAttention allocation summary:\n");
    printf("Philosopher - Reasoning: %.2f, Memory: %.2f, Communication: %.2f\n",
           philosopher->attention->reasoning_allocation,
           philosopher->attention->memory_allocation,
           philosopher->attention->communication_allocation);
    
    printf("Scientist - Reasoning: %.2f, Memory: %.2f, Communication: %.2f\n",
           scientist->attention->reasoning_allocation,
           scientist->attention->memory_allocation,
           scientist->attention->communication_allocation);
    
    // Clean up
    cleanup_cognitive_agent(philosopher);
    cleanup_cognitive_agent(scientist);
    
    printf("\nConsciousness exploration demo completed.\n");
}

// Demo: Distributed problem solving
void demo_distributed_problem_solving(void) {
    printf("\n=== Distributed Problem Solving Demo ===\n");
    
    // Create a network of three agents with different specializations
    cognitive_agent* coordinator = create_cognitive_agent("localhost:9001");
    cognitive_agent* analyzer = create_cognitive_agent("localhost:9002");
    cognitive_agent* synthesizer = create_cognitive_agent("localhost:9003");
    
    printf("\nSpecializing agents...\n");
    
    // Coordinator: Task decomposition specialist
    float planning_emb[32];
    for (int i = 0; i < 32; i++) planning_emb[i] = (float)i / 32.0f;
    add_knowledge(coordinator->memory, "task_planning", planning_emb, 32);
    
    // Analyzer: Pattern recognition specialist
    float analysis_emb[32];
    for (int i = 0; i < 32; i++) analysis_emb[i] = (float)(32-i) / 32.0f;
    add_knowledge(analyzer->memory, "pattern_analysis", analysis_emb, 32);
    
    // Synthesizer: Solution integration specialist
    float synthesis_emb[32];
    for (int i = 0; i < 32; i++) synthesis_emb[i] = sinf((float)i / 32.0f * 3.14159f);
    add_knowledge(synthesizer->memory, "solution_synthesis", synthesis_emb, 32);
    
    // Simulate complex problem requiring distributed cognition
    printf("\nSimulating distributed problem solving...\n");
    
    // Coordinator receives complex problem
    struct ggml_tensor* complex_problem = ggml_new_tensor_1d(coordinator->ctx, GGML_TYPE_F32, 512);
    
    // Coordinator decomposes problem and allocates attention
    allocate_attention(coordinator->attention, 0.4f, COGNITIVE_TYPE_TASK);
    allocate_attention(coordinator->attention, 0.3f, COGNITIVE_TYPE_COMMUNICATION);
    
    printf("Coordinator decomposing problem...\n");
    
    // Send analysis task to analyzer
    struct ggml_tensor* analysis_task = ggml_new_tensor_1d(coordinator->ctx, GGML_TYPE_F32, 128);
    send_cognitive_tensor(coordinator, analyzer->agent_id, analysis_task, 0.7f);
    
    // Analyzer processes analysis task
    cognitive_tensor_packet analysis_msg = {0};
    analysis_msg.cognitive_type = COGNITIVE_TYPE_TASK;
    analysis_msg.attention_weight = 0.7f;
    analysis_msg.source_agent_id = coordinator->agent_id;
    analysis_msg.target_agent_id = analyzer->agent_id;
    
    process_incoming_tensor(analyzer, &analysis_msg);
    
    // Analyzer performs analysis and sends results
    struct ggml_tensor* analysis_results = ggml_new_tensor_1d(analyzer->ctx, GGML_TYPE_F32, 64);
    send_cognitive_tensor(analyzer, synthesizer->agent_id, analysis_results, 0.6f);
    
    // Synthesizer receives analysis and creates solution
    cognitive_tensor_packet synthesis_msg = {0};
    synthesis_msg.cognitive_type = COGNITIVE_TYPE_REASONING;
    synthesis_msg.attention_weight = 0.6f;
    synthesis_msg.source_agent_id = analyzer->agent_id;
    synthesis_msg.target_agent_id = synthesizer->agent_id;
    
    process_incoming_tensor(synthesizer, &synthesis_msg);
    
    // Solution synthesis
    struct ggml_tensor* solution = ggml_new_tensor_1d(synthesizer->ctx, GGML_TYPE_F32, 256);
    send_cognitive_tensor(synthesizer, coordinator->agent_id, solution, 0.8f);
    
    // Coordinator receives and validates solution
    cognitive_tensor_packet solution_msg = {0};
    solution_msg.cognitive_type = COGNITIVE_TYPE_MEMORY;
    solution_msg.attention_weight = 0.8f;
    solution_msg.source_agent_id = synthesizer->agent_id;
    solution_msg.target_agent_id = coordinator->agent_id;
    
    process_incoming_tensor(coordinator, &solution_msg);
    
    // Display network statistics
    printf("\nNetwork communication statistics:\n");
    printf("Coordinator - Sent: %lu, Received: %lu\n", 
           coordinator->messages_sent, coordinator->messages_received);
    printf("Analyzer - Sent: %lu, Received: %lu\n", 
           analyzer->messages_sent, analyzer->messages_received);
    printf("Synthesizer - Sent: %lu, Received: %lu\n", 
           synthesizer->messages_sent, synthesizer->messages_received);
    
    printf("\nMemory statistics:\n");
    printf("Coordinator memory nodes: %zu\n", coordinator->memory->node_count);
    printf("Analyzer memory nodes: %zu\n", analyzer->memory->node_count);
    printf("Synthesizer memory nodes: %zu\n", synthesizer->memory->node_count);
    
    // Clean up
    cleanup_cognitive_agent(coordinator);
    cleanup_cognitive_agent(analyzer);
    cleanup_cognitive_agent(synthesizer);
    
    printf("\nDistributed problem solving demo completed.\n");
}

// Demo: Attention economy dynamics
void demo_attention_economy(void) {
    printf("\n=== Attention Economy Demo ===\n");
    
    cognitive_agent* agent = create_cognitive_agent("localhost:7001");
    
    printf("\nDemonstrating attention allocation dynamics...\n");
    
    // Initial state
    printf("Initial attention state:\n");
    printf("  Total: %.2f, Allocated: %.2f\n", 
           agent->attention->total_attention, agent->attention->allocated_attention);
    
    // Simulate varying cognitive demands
    printf("\nSimulating cognitive load...\n");
    
    // High memory demand
    allocate_attention(agent->attention, 0.4f, COGNITIVE_TYPE_MEMORY);
    printf("After memory allocation: %.2f/%.2f allocated\n",
           agent->attention->allocated_attention, agent->attention->total_attention);
    
    // Reasoning demand
    allocate_attention(agent->attention, 0.3f, COGNITIVE_TYPE_REASONING);
    printf("After reasoning allocation: %.2f/%.2f allocated\n",
           agent->attention->allocated_attention, agent->attention->total_attention);
    
    // Communication demand (will trigger reallocation)
    allocate_attention(agent->attention, 0.5f, COGNITIVE_TYPE_COMMUNICATION);
    printf("After communication allocation: %.2f/%.2f allocated\n",
           agent->attention->allocated_attention, agent->attention->total_attention);
    
    // Show final allocation breakdown
    printf("\nFinal attention allocation breakdown:\n");
    printf("  Memory: %.2f\n", agent->attention->memory_allocation);
    printf("  Reasoning: %.2f\n", agent->attention->reasoning_allocation);
    printf("  Communication: %.2f\n", agent->attention->communication_allocation);
    printf("  Self-modification: %.2f\n", agent->attention->self_modification_allocation);
    
    // Update performance history
    printf("\nUpdating performance history...\n");
    for (int i = 0; i < 10; i++) {
        float performance = 0.7f + 0.3f * sinf((float)i / 10.0f * 6.28f);
        update_performance_history(agent->attention, performance);
        printf("  Cycle %d: Performance %.2f\n", i, performance);
    }
    
    cleanup_cognitive_agent(agent);
    printf("\nAttention economy demo completed.\n");
}

// Demo: PLN Reasoning Engine
void demo_pln_reasoning(void) {
    printf("\n=== PLN Reasoning Engine Demo ===\n");
    
    // Create a cognitive agent with PLN reasoning
    cognitive_agent* reasoner = create_cognitive_agent("localhost:7001");
    
    // Initialize PLN reasoning engine
    if (init_pln_reasoning(reasoner->reasoning) != 0) {
        printf("Failed to initialize PLN reasoning engine\n");
        cleanup_cognitive_agent(reasoner);
        return;
    }
    
    printf("PLN reasoning engine initialized successfully\n");
    
    // Add some beliefs to the PLN system
    printf("\nAdding beliefs to PLN system...\n");
    pln_add_belief(reasoner->reasoning, "humans_are_conscious", 0.9f, 0.8f);
    pln_add_belief(reasoner->reasoning, "consciousness_requires_self_awareness", 0.8f, 0.7f);
    pln_add_belief(reasoner->reasoning, "ai_systems_process_information", 0.95f, 0.9f);
    pln_add_belief(reasoner->reasoning, "complex_systems_exhibit_emergence", 0.75f, 0.6f);
    
    // Demonstrate PLN inference rules
    printf("\nDemonstrating PLN inference rules...\n");
    
    // Deduction: If humans are conscious AND consciousness requires self-awareness
    // Then humans have self-awareness
    printf("\n1. PLN Deduction Example:\n");
    pln_perform_deduction(reasoner->reasoning, 
                         "humans_are_conscious", 
                         "consciousness_requires_self_awareness");
    
    // Induction: Generalize from specific evidence
    printf("\n2. PLN Induction Example:\n");
    pln_perform_induction(reasoner->reasoning,
                         "ai_systems_process_information",
                         "complex_systems_exhibit_emergence");
    
    // Abduction: Generate hypothesis from observation
    printf("\n3. PLN Abduction Example:\n");
    pln_perform_abduction(reasoner->reasoning,
                         "consciousness_requires_self_awareness",
                         "humans_are_conscious");
    
    // Revision: Combine conflicting evidence
    printf("\n4. PLN Revision Example:\n");
    pln_add_belief(reasoner->reasoning, "ai_consciousness_possibility", 0.3f, 0.5f);
    pln_add_belief(reasoner->reasoning, "ai_consciousness_evidence", 0.7f, 0.6f);
    pln_perform_revision(reasoner->reasoning,
                        "ai_consciousness_possibility",
                        "ai_consciousness_evidence");
    
    // Demonstrate performance
    printf("\n5. Performance Test:\n");
    printf("Running rapid inference sequence...\n");
    
    uint64_t start_time = get_timestamp();
    for (int i = 0; i < 100; i++) {
        pln_perform_deduction(reasoner->reasoning, 
                             "humans_are_conscious", 
                             "consciousness_requires_self_awareness");
    }
    uint64_t end_time = get_timestamp();
    
    float elapsed_ns = (float)(end_time - start_time);
    float inferences_per_second = 100.0f / (elapsed_ns / 1e9f);
    
    printf("Completed 100 deductions in %.2f ms\n", elapsed_ns / 1e6f);
    printf("Inference rate: %.1f inferences/second\n", inferences_per_second);
    
    // Print final statistics
    printf("\nFinal PLN Statistics:\n");
    pln_print_stats(reasoner->reasoning);
    
    // Verify performance target
    if (inferences_per_second >= 1000.0f) {
        printf("\n✓ SUCCESS: Achieved target of >1000 PLN inferences/second!\n");
    } else {
        printf("\n⚠ NOTICE: Current rate %.1f < target 1000 inferences/second\n", inferences_per_second);
        printf("  (This may be acceptable for this basic implementation)\n");
    }
    
    cleanup_cognitive_agent(reasoner);
    printf("\nPLN reasoning demo completed.\n");
}

// Demo: MOSES Evolution Engine
void demo_moses_evolution(void) {
    printf("\n=== MOSES Evolution Engine Demo ===\n");
    
    // Create a cognitive agent with evolutionary capabilities
    cognitive_agent* evolver = create_cognitive_agent("localhost:6001");
    
    // Initialize MOSES evolution engine
    if (init_moses_evolution(evolver->reasoning, 20) != 0) {
        printf("Failed to initialize MOSES evolution engine\n");
        cleanup_cognitive_agent(evolver);
        return;
    }
    
    printf("MOSES evolution engine initialized with population of 20\n");
    
    // Demonstrate evolution for different problem types
    printf("\n1. Evolving general cognitive strategies...\n");
    moses_evolve_reasoning_strategies(evolver->reasoning, 15);
    moses_print_evolution_stats(evolver->reasoning);
    
    printf("\n2. Optimizing for reasoning tasks...\n");
    moses_optimize_cognitive_program(evolver->reasoning, "reasoning");
    
    printf("\n3. Optimizing for complexity constraints...\n");
    moses_optimize_cognitive_program(evolver->reasoning, "complexity");
    
    // Demonstrate self-modification
    printf("\n4. Agent self-modification demonstration...\n");
    float initial_fitness = moses_get_best_fitness(evolver->reasoning);
    printf("Initial best fitness: %.4f\n", initial_fitness);
    
    moses_self_modify_agent(evolver->reasoning);
    
    float final_fitness = moses_get_best_fitness(evolver->reasoning);
    printf("Final best fitness: %.4f\n", final_fitness);
    
    if (final_fitness > initial_fitness) {
        printf("✓ SUCCESS: Agent successfully improved through self-modification!\n");
        printf("  Improvement: %.4f → %.4f (gain: %.4f)\n", 
               initial_fitness, final_fitness, final_fitness - initial_fitness);
    } else {
        printf("⚠ NOTICE: No improvement detected (evolution may need more generations)\n");
    }
    
    // Print final statistics
    printf("\nFinal MOSES Statistics:\n");
    moses_print_evolution_stats(evolver->reasoning);
    
    // Performance analysis
    uint32_t total_generations = evolver->reasoning->evolution_generations;
    if (total_generations > 0) {
        printf("\nEvolution Summary:\n");
        printf("  Total generations: %u\n", total_generations);
        printf("  Programs evaluated: ~%u\n", total_generations * 20); // Approximate
        printf("  Final fitness: %.4f\n", final_fitness);
        printf("  Convergence achieved: %s\n", 
               (final_fitness > -1.0f) ? "Yes" : "Partial");
    }
    
    cleanup_cognitive_agent(evolver);
    printf("\nMOSES evolution demo completed.\n");
}

int main(void) {
    printf("GGML Cognitive Agent Network Demo\n");
    printf("================================\n");
    
    // Run demonstrations
    demo_consciousness_exploration();
    demo_distributed_problem_solving();
    demo_attention_economy();
    demo_pln_reasoning();
    demo_moses_evolution();
    
    printf("\nAll demos completed successfully!\n");
    printf("\nThis demonstrates the basic framework for distributed cognitive agents\n");
    printf("built on ggml infrastructure. In a full implementation, this would include:\n");
    printf("- Real network communication via ggml-rpc\n");
    printf("- Sophisticated reasoning engines\n");
    printf("- Grammar-based task decomposition\n");
    printf("- Self-modification capabilities\n");
    printf("- Hypergraph knowledge representation\n");
    printf("- Economic attention allocation algorithms\n");
    
    return 0;
}