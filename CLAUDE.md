# DSPy + LangGraph Integration Test

## Project Overview

This repository tests integrating **DSPy** (declarative prompt optimization) with **LangGraph** (multi-agent orchestration) using the **DeepAgents CLI** framework.

## What We're Testing

### DSPy
- Framework for programming language models declaratively
- Automatic prompt optimization based on data and metrics
- Eliminates manual prompt engineering

### LangGraph
- Framework for building multi-agent applications as graphs
- Handles agent workflows and state management
- Provides control over agent coordination

### DeepAgents CLI
- LangChain's CLI tool for building agents with persistent memory
- Provides middleware, tools, and CLI interface

## Integration Approach

```
User Input
    ↓
LangGraph Orchestration (routing, state management)
    ↓
Sub-Agents (LangGraph nodes)
    ↓
DSPy Optimization (inside each agent)
    ↓
LLM Providers
```

**Architecture**: LangGraph handles orchestration, DSPy lives inside individual agents for prompt optimization.

## Documentation

- **[DSPy Overview](docs/dspy-overview.md)**: DSPy framework, optimizers, and usage
- **[LangGraph & DeepAgents](docs/langgraph-deepagents.md)**: LangGraph architecture and DeepAgents CLI
- **[Integration Plan](docs/integration-plan.md)**: Implementation plan with code examples

## Current Files

- `simple_agent.py`: Basic DeepAgents example
- `cli.py`: Interactive CLI interface
- `README.md`: Setup instructions

## Next Steps

1. Implement DSPy + LangGraph integration from the plan
2. Create example agents with DSPy optimization
3. Build CLI interface demonstrating the integration
4. Test optimization capabilities

## Why This Integration

**Without integration**: Manual prompt engineering, static prompts, model-specific optimization

**With integration**: Automatic prompt optimization, model-agnostic agents, learning from interactions

## Status

- **Research & Documentation**: Complete
- **Implementation**: Pending
- **Testing**: Pending