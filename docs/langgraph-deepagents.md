# LangGraph and DeepAgents CLI: Building Complex Agentic CLI Workflows

## DeepAgents CLI Overview

DeepAgents CLI is LangChain's command-line tool for coding, research, and building agents with persistent memory. Introduced in November 2025, it provides a terminal-based interface for creating and running custom DeepAgents, positioning itself as "a general purpose version of Claude Code."

### Key Features

- **Multi-Model Support**: Works with both Anthropic (Claude) and OpenAI models
- **Default Model**: Anthropic Claude Sonnet 4 
- **Web Search**: Integrated Tavily for web search capabilities
- **Persistent Memory**: Knowledge retention across sessions
- **File Operations**: Read, write, and edit files within projects
- **Shell Commands**: Execute commands with human approval
- **API Integration**: Make HTTP requests to external APIs

### Installation

```bash
# Install via pip
pip install deepagents

# Or using uv
uv add deepagents

# Or using poetry
poetry add deepagents
```

### CLI Commands

The DeepAgents CLI provides several commands for agent management:

```bash
# List existing agents
deepagents list

# Create a new specialized agent
deepagents create [AGENT_NAME]

# Use a specific agent
deepagents use [AGENT_NAME]

# Reset an agent to default state
deepagents reset [AGENT_NAME]

# Run the CLI interface
deepagents run
```

### Persistent Memory System

#### Memory-First Protocol
DeepAgents follows a systematic approach to knowledge management:
1. **Check Memory First**: Always searches `/memories/` for relevant knowledge
2. **Search Before Answering**: Looks through memory files when uncertain
3. **Save New Information**: Automatically stores learned information to `/memories/`

#### Storage Location
- Agent memories stored in: `~/.deepagents/AGENT_NAME/memories/`
- Default agent name: `"agent"`
- Each agent maintains isolated knowledge base

### Configuration

DeepAgents CLI can be configured through environment variables:

```bash
# Set the LLM provider
export DEEPAGENTS_MODEL="anthropic"  # or "openai"

# Set API keys
export ANTHROPIC_API_KEY="your_key"
export OPENAI_API_KEY="your_key"

# Configure Tavily for web search
export TAVILY_API_KEY="your_key"
```

### Creating Specialized Agents

You can create agents for different purposes:

```bash
# Create a research agent
deepagents create research_agent

# Create a coding assistant
deepagents create code_helper

# Create a data analysis agent
deepagents create data_analyst
```

Each agent maintains its own:
- Memory database
- Configuration settings
- Tool permissions
- Conversation history

## LangGraph Overview

LangGraph is a Python framework for building stateful, multi-agent applications with Large Language Models (LLMs). It models agent workflows as graphs, enabling complex control flows, conditional logic, and multi-agent coordination.

### Core Architecture

#### Graph-Based Model
LangGraph represents workflows as directed graphs where:
- **Nodes**: Python functions that encode agent logic
- **Edges**: Control flow between nodes (conditional or fixed)
- **State**: Shared data structure representing application snapshot

#### Key Components

1. **StateGraph**: The main container for your application
2. **Nodes**: Individual processing units (agents, tools, logic)
3. **Edges**: Connections defining execution flow
4. **Checkpointer**: Persistence layer for state management
5. **Compiled Graph**: The executable workflow

### Building Blocks

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from operator import add

class State(TypedDict):
    messages: Annotated[list, add]
    
graph = StateGraph(State)
graph.add_node("agent", agent_function)
graph.add_edge("agent", END)
app = graph.compile()
```

## DeepAgents Python Library

DeepAgents is a Python library built on top of LangGraph that implements advanced agent architectures for complex, multi-step tasks. It provides pre-built middleware components that add sophisticated capabilities to base LangGraph agents.

### Architecture Layers

1. **Base Layer**: LangGraph CompiledStateGraph
2. **Middleware Layer**: Composable capabilities
3. **Application Layer**: Your custom agent logic

### Core Middleware Components

#### 1. TodoListMiddleware
Provides task management capabilities:
- `write_todos` tool for maintaining task lists
- Automatic prompting for multi-part task tracking
- Progress visibility and organization

```python
from deepagents.middlewares import TodoListMiddleware

middleware = TodoListMiddleware()
```

#### 2. FilesystemMiddleware
Virtual filesystem in agent state:
- Single-level file system (no subdirectories)
- Stored entirely in LangGraph state
- Tools: `write_file`, `read_file`, `list_files`, `delete_file`

```python
from deepagents.middlewares import FilesystemMiddleware

middleware = FilesystemMiddleware()
```

#### 3. SubAgentMiddleware
Hierarchical agent architectures:
- `task` tool for delegating to sub-agents
- Isolated context for each sub-agent
- Different tools/instructions per sub-agent

```python
from deepagents.middlewares import SubAgentMiddleware

middleware = SubAgentMiddleware()
```

### Creating a DeepAgent Programmatically

```python
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")

agent = create_deep_agent(
    model=llm,
    system_prompt="You are a helpful assistant.",
    middlewares=[...]  # Optional custom middleware
)

# The returned agent is a standard LangGraph CompiledStateGraph
result = agent.invoke({"messages": [...]})
```

## Multi-Agent Workflow Patterns

### 1. Supervisor Pattern
A supervisor agent coordinates multiple specialized agents:

```python
class SupervisorState(TypedDict):
    messages: list
    next_agent: str
    
supervisor_graph = StateGraph(SupervisorState)
supervisor_graph.add_node("supervisor", supervisor_agent)
supervisor_graph.add_node("researcher", research_agent)
supervisor_graph.add_node("coder", coding_agent)
```

### 2. Network Pattern
Agents communicate directly in a network topology:
- Each agent is a graph node
- Edges represent communication channels
- State propagates through the network

### 3. Hierarchical Pattern
Nested agent structures with delegation:
- Main agent handles high-level logic
- Sub-agents handle specialized tasks
- Results bubble up through hierarchy

## State Management and Persistence

### Checkpointing
LangGraph provides built-in persistence:
```python
from langgraph.checkpoint import MemorySaver

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)
```

### Memory Features
- Conversation history persistence
- Cross-session context maintenance
- Time-travel debugging capabilities

## Human-in-the-Loop (HITL)

### Interrupts and Approvals
Configure tool-level interrupts:
```python
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["dangerous_tool"]
)
```

### Review and Rollback
- Inspect agent actions before execution
- Roll back and take different actions
- Manual intervention points

## CLI Workflow Implementation Examples

### Example 1: Simple Research Assistant CLI

```python
#!/usr/bin/env python3
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI

def create_research_agent():
    llm = ChatOpenAI(model="gpt-4")
    
    agent = create_deep_agent(
        model=llm,
        system_prompt="""You are a research assistant. 
        Use the Task tool for complex research tasks."""
    )
    
    return agent

def main():
    agent = create_research_agent()
    
    while True:
        user_input = input("> ")
        if user_input.lower() == 'quit':
            break
            
        result = agent.invoke({
            "messages": [{"role": "user", "content": user_input}]
        })
        
        print(result["messages"][-1].content)

if __name__ == "__main__":
    main()
```

### Example 2: Multi-Agent Development Team

```python
from deepagents import create_deep_agent
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class TeamState(TypedDict):
    messages: List[dict]
    current_task: str
    completed_tasks: List[str]

def create_development_team():
    # Create specialized agents
    architect = create_deep_agent(
        model=llm,
        system_prompt="You are a software architect. Design systems and APIs."
    )
    
    developer = create_deep_agent(
        model=llm,
        system_prompt="You are a developer. Write clean, efficient code."
    )
    
    tester = create_deep_agent(
        model=llm,
        system_prompt="You are a QA engineer. Write tests and find bugs."
    )
    
    # Create team workflow graph
    workflow = StateGraph(TeamState)
    workflow.add_node("architect", architect)
    workflow.add_node("developer", developer)
    workflow.add_node("tester", tester)
    
    # Define workflow logic
    workflow.add_edge("architect", "developer")
    workflow.add_edge("developer", "tester")
    workflow.add_edge("tester", END)
    
    return workflow.compile()
```

## Advanced Features

### 1. Subgraph Architecture
Group related agents into reusable components:
```python
subgraph = StateGraph(SubState)
# ... define subgraph nodes and edges
main_graph.add_node("subgraph", subgraph.compile())
```

### 2. Streaming Support
Real-time output streaming:
```python
async for chunk in app.astream(input_state):
    print(chunk)
```

### 3. LangGraph Studio
Visual debugging and development:
- Desktop app for MacOS
- Web-based development server
- Visual graph representation
- Step-by-step debugging

## Integration Points

### With Cloud Services
- **AWS Bedrock**: Scalable model deployment
- **LangSmith**: Observability and monitoring
- **Vector Stores**: For RAG implementations

### With Other Frameworks
- **LangChain**: Seamless integration as base framework
- **OpenAI**: Direct API compatibility
- **Custom Tools**: Easy tool integration

## Best Practices

### 1. State Design
- Keep state minimal and focused
- Use TypedDict for type safety
- Consider state size for checkpointing

### 2. Graph Structure
- Start simple, add complexity gradually
- Use conditional edges sparingly
- Document complex control flows

### 3. Error Handling
- Implement retry logic in nodes
- Use fallback paths for failures
- Log errors for debugging

### 4. Performance
- Minimize state updates
- Use async where possible
- Consider parallel node execution

### 5. Memory Management
- Leverage DeepAgents CLI's persistent memory
- Organize memories by topic/domain
- Regularly review and prune outdated memories

## DeepAgents as an "Agent Harness"

LangChain positions DeepAgents not just as a framework or runtime, but as an "agent harness" - a complete system that:
- **Framework**: Provides the underlying structure (via LangGraph)
- **Runtime**: Executes and manages agent lifecycles
- **Harness**: Adds opinionated defaults, prompts, and tool handling

This makes DeepAgents particularly suitable for:
- Rapid prototyping of agent systems
- Building production-ready CLI applications
- Creating specialized agents with minimal boilerplate

## Resources and Documentation

### Official Resources
- **LangGraph GitHub**: https://github.com/langchain-ai/langgraph
- **DeepAgents GitHub**: https://github.com/langchain-ai/deepagents
- **LangChain Website**: https://www.langchain.com/langgraph
- **DeepAgents CLI Blog**: https://blog.langchain.com/introducing-deepagents-cli/

### Documentation
- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **DeepAgents Overview**: https://docs.langchain.com/oss/python/deepagents/overview
- **Multi-Agent Guide**: https://blog.langchain.com/langgraph-multi-agent-workflows/

### Tutorials
- **Building Multi-Agent Systems**: Various Medium articles
- **LangGraph 101**: https://towardsdatascience.com/langgraph-101-lets-build-a-deep-research-agent/
- **AWS Integration**: https://aws.amazon.com/blogs/machine-learning/build-multi-agent-systems-with-langgraph-and-amazon-bedrock/

## Conclusion

LangGraph and DeepAgents CLI together provide a comprehensive solution for building complex, stateful, multi-agent CLI applications. The graph-based architecture offers precise control over agent workflows, while DeepAgents' middleware system and CLI interface add sophisticated capabilities like persistent memory, task management, and hierarchical agent structures. This combination enables rapid development of production-ready agentic systems that can handle complex, real-world tasks while maintaining knowledge across sessions.