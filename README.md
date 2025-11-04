# Math Assistant Prototype

A simple demonstration of **Development-Time DSPy + Production DeepAgents CLI** integration. This shows how to optimize prompts with DSPy during development, then deploy those optimized prompts with DeepAgents CLI for production use.

## Quick Start

### 1. Setup
```bash
# Install dependencies
uv sync

# Set your OpenAI API key
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### 2. Optimize Prompts
```bash
# Optimize prompts for GPT-4 (takes a few minutes)
uv run optimize-prompts --model openai/gpt-4
```

### 3. Use the Math Assistant
```bash
# Single calculation
uv run math-cli --problem "What is 25 + 17 - 9?"

# Interactive mode
uv run math-cli
```

## How It Works

This prototype demonstrates a **two-stage approach**:

1. **Development Time**: DSPy optimizes prompts using training data
2. **Production Time**: DeepAgents CLI uses the optimized prompts

### The Workflow

```
Training Data → DSPy Optimization → Optimized Prompts → DeepAgents CLI → User
```

**Benefits**: Get DSPy's automatic prompt optimization + DeepAgents' production tooling without runtime complexity.

## Commands

### Optimization
```bash
# Optimize for different models
uv run optimize-prompts --model openai/gpt-4
uv run optimize-prompts --model openai/gpt-3.5-turbo

# Check what was optimized
ls optimized_prompts/
```

### Math Assistant
```bash
# Interactive mode
uv run math-cli

# Single problem
uv run math-cli --problem "Calculate 100 - 45 + 30"

# Use different model (after optimizing for it)
uv run math-cli --model gpt-3.5-turbo
```

### Available Commands in Interactive Mode
- `help` - Show available commands  
- `info` - Show optimization information
- `test` - Run a quick test calculation
- `quit` - Exit the application

## Example Session

```bash
$ uv run math-cli

🧮 Math Assistant CLI
Powered by DSPy-optimized prompts + DeepAgents
📊 Using optimized prompts (score: 0.850)
🤖 Model: gpt-4
--------------------------------------------------

> What is 15 + 27?

🤔 Thinking about: What is 15 + 27?
----------------------------------------
🔧 TOOL CALLED: add(15.0, 27.0) = 42.0
🧮 The sum of 15 and 27 is 42.

> Calculate 50 - 23 + 10

🤔 Thinking about: Calculate 50 - 23 + 10
----------------------------------------
🔧 TOOL CALLED: subtract(50.0, 23.0) = 27.0
🔧 TOOL CALLED: add(27.0, 10.0) = 37.0
🧮 Let me solve this step by step:
First: 50 - 23 = 27
Then: 27 + 10 = 37
Therefore: 50 - 23 + 10 = 37
```

## File Structure

```
├── pyproject.toml          # Project configuration
├── .env                    # Your API key
├── datasets/
│   ├── examples.json       # Training data for optimization
│   └── metrics.py          # Evaluation metrics
├── math_tools.py          # Addition/subtraction tools
├── optimizer.py           # DSPy optimization logic
├── math_cli.py           # Main CLI application
├── agent_factory.py      # Loads optimized prompts into DeepAgents
├── optimized_prompts/     # Generated optimized prompts
└── docs/                  # Technical documentation
```

## Understanding the Integration

### DSPy (Development Time)
- **Input**: Training examples in `datasets/examples.json`
- **Process**: Automatic prompt optimization using MIPROv2
- **Output**: Optimized prompts saved to `optimized_prompts/`

### DeepAgents CLI (Production Time)  
- **Input**: Optimized prompts from DSPy
- **Process**: Creates production CLI with tools and middleware
- **Output**: Interactive math assistant

### Why This Approach?
- **No Runtime Complexity**: DSPy optimization happens offline
- **Production Ready**: Full DeepAgents tooling (file operations, task management)
- **Model Agnostic**: Easy model switching by re-running optimization
- **Best of Both**: DSPy's optimization + DeepAgents' orchestration

## Extending the Prototype

### Add New Math Operations

1. **Add tools** in `math_tools.py`:
```python
@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    result = a * b
    print(f"🔧 TOOL CALLED: multiply({a}, {b}) = {result}")
    return result

# Update get_langchain_tools()
def get_langchain_tools():
    return [add, subtract, multiply]
```

2. **Add training examples** in `datasets/examples.json`:
```json
{
  "problem": "What is 6 × 7?",
  "reasoning": "I need to multiply 6 and 7 using the multiplication tool.",
  "tool_calls": [{"tool": "multiply", "args": [6, 7]}],
  "final_answer": "6 × 7 = 42"
}
```

3. **Re-optimize**:
```bash
uv run optimize-prompts --model openai/gpt-4
```

### Switch Models

```bash
# Optimize for Claude (if you have Anthropic API key)
uv run optimize-prompts --model anthropic/claude-3-sonnet-20240229

# Use the optimized Claude model
uv run math-cli --model claude-3-sonnet-20240229
```

### Customize Training Data

Edit `datasets/examples.json` to add more complex problems, different reasoning styles, or domain-specific examples.

## Technical Details

For deeper technical information, see the `docs/` directory:
- **[DSPy Overview](docs/dspy-overview.md)**: Complete DSPy guide
- **[LangGraph & DeepAgents](docs/langgraph-deepagents.md)**: DeepAgents architecture
- **[Integration Plans](docs/)**: Alternative approaches and comparisons

## Troubleshooting

### "No optimized prompt found"
```bash
# Run optimization first
uv run optimize-prompts --model openai/gpt-4
```

### "Invalid API key"
```bash
# Check your .env file has the correct key
cat .env
# Should show: OPENAI_API_KEY=sk-...
```

### "Optimization failed"
- Verify your API key works
- Check training data exists: `ls datasets/examples.json`
- Try with a smaller model: `--model openai/gpt-3.5-turbo`

### Tools not being called
- The tools should be called automatically
- Look for `🔧 TOOL CALLED:` messages in output
- If missing, the agent may be solving simple problems without tools

## Requirements

- **Python**: 3.11+
- **API Key**: OpenAI API key (or other supported providers)
- **Tools**: uv (for dependency management)