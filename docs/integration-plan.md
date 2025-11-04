# DSPy + LangGraph Integration Plan

## Executive Summary

This document outlines the architectural plan for integrating DSPy's declarative prompt optimization with LangGraph's multi-agent orchestration capabilities in the DeepAgents CLI framework. The goal is to create LangGraph agents that leverage DSPy for automatic prompt optimization, reducing manual prompt engineering while maintaining the flexibility and control of graph-based workflows.

## Integration Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────┐
│                  User Interface                  │
│              (DeepAgents CLI/API)               │
└─────────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────────┐
│              LangGraph Orchestration             │
│         (Graph-based workflow control)           │
└─────────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────────┐
│               DSPy-Wrapped Agents                │
│        (Optimized prompts & reasoning)           │
└─────────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────────┐
│                   LLM Provider                   │
│          (OpenAI, Anthropic, Llama, etc.)        │
└─────────────────────────────────────────────────┘
```

### Component Roles

1. **LangGraph**: Handles orchestration, state management, and control flow
2. **DSPy**: Manages prompt optimization, few-shot learning, and reasoning strategies
3. **DeepAgents**: Provides middleware, tools, and CLI interface

## Clean Architecture: Simple Multi-Agent Example

### Layer Separation

The architecture has three clear layers:
1. **LangGraph handles orchestration** (main agent + routing)
2. **Sub-agents are LangGraph nodes**
3. **DSPy lives INSIDE each sub-agent** for prompt optimization

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import dspy

# ============================================
# LAYER 1: LangGraph State and Orchestration
# ============================================

class WorkflowState(TypedDict):
    """Shared state for all agents"""
    messages: List[dict]
    current_task: str
    task_type: str  # "research" or "code"
    result: str

def route_task(state: WorkflowState) -> str:
    """Router function - decides which agent to use"""
    task = state["current_task"].lower()
    
    if "research" in task or "find" in task or "explain" in task:
        return "researcher"
    elif "code" in task or "implement" in task or "function" in task:
        return "coder"
    else:
        return "researcher"  # default

# ============================================
# LAYER 2: Sub-Agents (LangGraph Nodes)
# ============================================

class ResearchAgent:
    """Research agent with DSPy optimization inside"""
    
    def __init__(self):
        # DSPy setup - this is INTERNAL to the agent
        dspy.configure(lm=dspy.LM('openai/gpt-4'))
        
        # Define what this agent does using DSPy signature
        class ResearchTask(dspy.Signature):
            """Research a topic thoroughly"""
            question = dspy.InputField(desc="research question")
            context = dspy.InputField(desc="any relevant context")
            answer = dspy.OutputField(desc="comprehensive answer with sources")
        
        # Use DSPy's ChainOfThought for reasoning
        self.engine = dspy.ChainOfThought(ResearchTask)
    
    def __call__(self, state: WorkflowState) -> WorkflowState:
        """LangGraph calls this - it uses DSPy internally"""
        # Extract input from LangGraph state
        task = state["current_task"]
        context = " ".join([m["content"] for m in state["messages"]])
        
        # Use DSPy to process the task
        result = self.engine(question=task, context=context)
        
        # Update LangGraph state with result
        state["result"] = result.answer
        state["messages"].append({
            "role": "assistant",
            "content": f"Research completed: {result.answer}"
        })
        
        return state

class CodingAgent:
    """Coding agent with DSPy optimization inside"""
    
    def __init__(self):
        # DSPy setup
        dspy.configure(lm=dspy.LM('openai/gpt-4'))
        
        # Define the coding task signature
        class CodingTask(dspy.Signature):
            """Generate code based on requirements"""
            task = dspy.InputField(desc="what to implement")
            language = dspy.InputField(desc="programming language")
            code = dspy.OutputField(desc="complete, working code")
            explanation = dspy.OutputField(desc="brief explanation")
        
        # Use DSPy's ProgramOfThought for code generation
        self.engine = dspy.ProgramOfThought(CodingTask)
    
    def __call__(self, state: WorkflowState) -> WorkflowState:
        """LangGraph calls this - it uses DSPy internally"""
        task = state["current_task"]
        
        # Use DSPy to generate code
        result = self.engine(
            task=task,
            language="python"  # default to Python
        )
        
        # Update state
        state["result"] = f"```python\n{result.code}\n```\n\n{result.explanation}"
        state["messages"].append({
            "role": "assistant",
            "content": f"Code generated: {result.code[:100]}..."
        })
        
        return state

# ============================================
# LAYER 3: LangGraph Workflow Assembly
# ============================================

def create_multi_agent_system():
    """Assemble the complete system using LangGraph"""
    
    # Create the workflow
    workflow = StateGraph(WorkflowState)
    
    # Initialize agents (with DSPy inside)
    researcher = ResearchAgent()
    coder = CodingAgent()
    
    # Add nodes to the graph
    workflow.add_node("router", route_task)
    workflow.add_node("researcher", researcher)
    workflow.add_node("coder", coder)
    
    # Define the flow
    workflow.set_entry_point("router")
    workflow.add_conditional_edges(
        "router",
        lambda x: x["task_type"],
        {
            "research": "researcher",
            "code": "coder"
        }
    )
    workflow.add_edge("researcher", END)
    workflow.add_edge("coder", END)
    
    # Compile the workflow
    return workflow.compile()

# ============================================
# Simple CLI Interface
# ============================================

def main():
    """Simple CLI to interact with the system"""
    print("Multi-Agent System (LangGraph + DSPy)")
    print("Type 'quit' to exit")
    print("-" * 40)
    
    # Create the multi-agent system
    app = create_multi_agent_system()
    
    while True:
        user_input = input("\n> ").strip()
        
        if user_input.lower() == 'quit':
            break
        
        # Prepare initial state
        initial_state = WorkflowState(
            messages=[{"role": "user", "content": user_input}],
            current_task=user_input,
            task_type="",
            result=""
        )
        
        # Run the workflow
        final_state = app.invoke(initial_state)
        
        # Display result
        print(f"\n📊 Result:\n{final_state['result']}")

if __name__ == "__main__":
    main()
```

## How It Works - Clear Separation

### 1. LangGraph Level (Orchestration)
- Defines the workflow as a graph
- Routes tasks between agents
- Manages state flow
- **No DSPy code at this level**

### 2. Agent Level (Nodes)
- Each agent is a callable class
- Takes LangGraph state as input
- Returns updated state
- **DSPy lives INSIDE each agent**

### 3. DSPy Level (Prompt Optimization)
- Defines signatures for what each agent does
- Handles prompt optimization internally
- Can be optimized without changing LangGraph structure

## Benefits of This Architecture

- **Clean Separation**: LangGraph doesn't know about DSPy
- **Modular**: Agents can be swapped or modified independently
- **Optimizable**: Each agent can be optimized separately
- **Testable**: Each layer can be tested in isolation

## Adding Optimization

```python
# You can add optimization to any agent without changing the workflow
class OptimizableResearchAgent(ResearchAgent):
    def __init__(self):
        super().__init__()
        self.training_data = []
    
    def optimize(self):
        """Optimize the DSPy engine with collected data"""
        if len(self.training_data) >= 10:
            optimizer = dspy.MIPROv2()
            self.engine = optimizer.compile(
                self.engine,
                trainset=self.training_data
            )
    
    def collect_feedback(self, task, result, score):
        """Collect training data for optimization"""
        self.training_data.append({
            "question": task,
            "answer": result,
            "score": score
        })
```

## Integration with DeepAgents CLI

### Simple Integration Approach

```python
from deepagents import create_deep_agent

def create_dspy_enhanced_deepagent():
    """Create a DeepAgent that uses our multi-agent system"""
    
    # Create the multi-agent system
    multi_agent_app = create_multi_agent_system()
    
    # Create a DeepAgent that wraps it
    agent = create_deep_agent(
        model=ChatOpenAI(model="gpt-4"),
        system_prompt="""You coordinate specialized agents.
        For research tasks, delegate to the research system.
        For coding tasks, delegate to the coding system.
        Use the Task tool to delegate complex work."""
    )
    
    return agent
```

### Advanced: Custom Middleware

```python
class DSPyMultiAgentMiddleware:
    """Middleware that adds multi-agent DSPy capabilities"""
    
    def __init__(self):
        self.multi_agent_system = create_multi_agent_system()
    
    def get_tools(self):
        """Add multi-agent tool to DeepAgent"""
        def delegate_to_specialist(task: str, task_type: str):
            """Delegate to specialized DSPy agents"""
            state = WorkflowState(
                messages=[{"role": "user", "content": task}],
                current_task=task,
                task_type=task_type,
                result=""
            )
            result = self.multi_agent_system.invoke(state)
            return result["result"]
        
        return [delegate_to_specialist]
```

## Use Cases

### 1. Simple Research + Code CLI
- User asks research question → goes to DSPy research agent
- User asks for code → goes to DSPy coding agent
- LangGraph handles routing and state management

### 2. Complex Multi-Step Tasks
- Break down complex tasks into subtasks
- Route each subtask to appropriate DSPy-optimized agent
- Combine results in LangGraph orchestration layer

### 3. Self-Improving System
- Each DSPy agent collects interaction data
- Periodic optimization improves agent performance
- LangGraph workflow remains unchanged

## Implementation Roadmap

### Phase 1: Basic Implementation
- [ ] Implement simple ResearchAgent with DSPy
- [ ] Implement simple CodingAgent with DSPy
- [ ] Create basic LangGraph workflow
- [ ] Build simple CLI interface

### Phase 2: DeepAgents Integration
- [ ] Create DSPy middleware for DeepAgents
- [ ] Integrate with existing DeepAgents CLI
- [ ] Add optimization capabilities

### Phase 3: Enhancement
- [ ] Add more specialized agents
- [ ] Implement automatic optimization
- [ ] Add monitoring and analytics

## Technical Considerations

### State Management
- Keep state simple and focused
- Ensure proper state flow between agents
- Consider state size for performance

### Error Handling
- Graceful fallback when DSPy optimization fails
- Proper error propagation through LangGraph
- Logging for debugging

### Performance
- Cache optimized prompts
- Consider async execution for parallel agents
- Monitor token usage during optimization

## Conclusion

This clean architecture provides:
- **Clear separation of concerns** between orchestration (LangGraph) and optimization (DSPy)
- **Modular design** that allows independent development and testing
- **Scalable approach** that can grow from simple to complex systems
- **Practical implementation** that integrates well with DeepAgents CLI

The key insight is keeping DSPy **internal** to each agent while letting LangGraph handle the **orchestration** at a higher level.

## Next Steps

1. Implement the basic example
2. Test with real scenarios
3. Integrate with DeepAgents CLI
4. Add optimization capabilities
5. Scale to more complex workflows

## References

- [DSPy Documentation](https://dspy.ai/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [DeepAgents CLI Blog](https://blog.langchain.com/introducing-deepagents-cli/)
- [Integration Examples](https://medium.com/@akankshasinha247/langgraph-dspy-orchestrating-multi-agent-ai-workflows-declarative-prompting-93b2bd06e995)