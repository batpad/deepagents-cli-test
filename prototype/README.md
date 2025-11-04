# Math Assistant Prototype

A simple demonstration of **Development-Time DSPy + Production DeepAgents CLI** integration.

This prototype shows how to:
1. **Optimize prompts** with DSPy during development
2. **Deploy optimized prompts** with DeepAgents CLI for production use
3. **Switch models** easily by re-running optimization

## Architecture

```
Development Time:
Golden Dataset → DSPy Optimization → Optimized Prompts → Save to JSON

Production Time:
User Input → DeepAgents CLI (with optimized prompts) → Math Tools → Results
```

## Features

- 🧮 **Simple Math Assistant**: Handles addition and subtraction with step-by-step reasoning
- 🔧 **DSPy Optimization**: Automatically optimizes prompts based on training data
- 🤖 **DeepAgents Integration**: Production-ready CLI with optimized prompts
- 🔄 **Model Switching**: Easy optimization for different models (GPT-4, GPT-3.5, etc.)
- 📊 **Performance Tracking**: Tracks optimization scores and improvements

## Quick Start

### 1. Setup

```bash
# Clone and navigate to prototype
cd prototype

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

### 2. Set API Key

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Optimize Prompts

```bash
# Optimize for GPT-4 (default)
uv run optimize-prompts

# Or optimize for specific model
uv run optimize-prompts --model openai/gpt-3.5-turbo
```

### 4. Use the Math Assistant

```bash
# Interactive mode
uv run math-cli

# Single calculation
uv run math-cli --problem "What is 25 + 17 - 9?"
```

## Usage Examples

### Interactive Mode

```bash
$ uv run math-cli

🧮 Math Assistant CLI
Powered by DSPy-optimized prompts + DeepAgents
📊 Using optimized prompts (score: 0.850)
🤖 Model: gpt-4
Type 'help' for commands, 'quit' to exit
--------------------------------------------------

> What is 15 + 27?

🤔 Thinking about: What is 15 + 27?
----------------------------------------
🧮 I need to add 15 and 27 together. Let me solve this step by step.

Using the addition tool: 15 + 27 = 42

Therefore, 15 + 27 = 42

> What's 50 - 23 + 10?

🤔 Thinking about: What's 50 - 23 + 10?
----------------------------------------
🧮 I need to solve this step by step:

First, I'll subtract: 50 - 23 = 27
Then, I'll add: 27 + 10 = 37

Therefore, 50 - 23 + 10 = 37
```

### Commands

- `help` - Show available commands
- `info` - Show optimization information
- `test` - Run a quick test calculation
- `quit` - Exit the application

## File Structure

```
prototype/
├── pyproject.toml              # Project configuration and dependencies
├── README.md                   # This file
├── datasets/
│   └── math_assistant/
│       ├── examples.json       # Training examples for optimization
│       └── metrics.py          # Evaluation metrics
├── src/
│   └── math_assistant/
│       ├── optimization/
│       │   ├── optimizer.py    # DSPy optimization logic
│       │   └── cli.py         # Optimization CLI
│       └── cli/
│           ├── math_tools.py   # Addition/subtraction functions
│           ├── agent_factory.py # DeepAgents integration
│           └── math_cli.py     # Interactive CLI
└── optimized_prompts/         # Generated optimized prompts
    └── math_assistant_openai_gpt-4.json
```

## How It Works

### 1. Golden Dataset

The training data in `datasets/math_assistant/examples.json` contains:

```json
[
  {
    "problem": "What is 15 + 27?",
    "reasoning": "I need to add 15 and 27 together. I'll use the addition tool to compute this.",
    "tool_calls": [{"tool": "add", "args": [15, 27]}],
    "final_answer": "15 + 27 = 42"
  }
]
```

### 2. DSPy Optimization

The optimizer:
- Loads training examples
- Creates DSPy signatures for math reasoning
- Uses MIPROv2 optimizer to find optimal prompts
- Evaluates performance with custom metrics
- Saves optimized prompts as JSON

### 3. DeepAgents Integration

The agent factory:
- Loads optimized prompts for the specified model
- Creates DeepAgents with optimized system prompts
- Integrates math tools (add, subtract)
- Provides fallback for unoptimized models

### 4. Model Switching

```bash
# Optimize for different models
uv run optimize-prompts --model openai/gpt-4
uv run optimize-prompts --model openai/gpt-3.5-turbo

# Use with specific model
uv run math-cli --model gpt-3.5-turbo
```

## Development Workflow

### Adding New Training Examples

Edit `datasets/math_assistant/examples.json`:

```json
{
  "problem": "Calculate 88 - 19",
  "reasoning": "I need to subtract 19 from 88 using the subtraction tool.",
  "tool_calls": [{"tool": "subtract", "args": [88, 19]}],
  "final_answer": "88 - 19 = 69"
}
```

### Customizing Metrics

Edit `datasets/math_assistant/metrics.py` to adjust evaluation criteria:

```python
def math_accuracy_metric(prediction, example):
    # Custom evaluation logic
    return score  # 0.0 to 1.0
```

### Testing Optimization

```bash
# Run optimization with verbose output
uv run optimize-prompts --model openai/gpt-4

# Compare multiple models
uv run optimize-prompts compare --models openai/gpt-4 openai/gpt-3.5-turbo
```

## Benefits of This Approach

### ✅ What We Get

1. **Model Agnostic**: Easy switching between models with re-optimization
2. **Production Ready**: Full DeepAgents CLI tooling
3. **Optimized Performance**: DSPy finds better prompts than manual engineering
4. **Clean Separation**: Development optimization vs production usage
5. **Maintainable**: Clear structure and responsibilities

### ⚠️ Trade-offs

1. **Two-Step Process**: Must optimize before using (not real-time learning)
2. **Model-Specific**: Need to re-optimize for each model
3. **Static Prompts**: No runtime adaptation (by design for reliability)

## Extending the Prototype

### Adding New Math Operations

1. **Add tools** in `src/math_assistant/cli/math_tools.py`:
   ```python
   def multiply(a, b):
       return a * b
   
   MATH_TOOLS["multiply"] = multiply
   ```

2. **Add training examples** with the new operation
3. **Re-run optimization** to incorporate new capabilities

### Adding More Complex Reasoning

1. **Create new DSPy signatures** for different problem types
2. **Extend the training dataset** with complex examples
3. **Adjust metrics** to evaluate complex reasoning

### Integration with Other Frameworks

The optimized prompts are saved as JSON and can be used with:
- Raw LangChain agents
- Other agent frameworks
- Custom implementations

## Troubleshooting

### Common Issues

**"No optimized prompt found"**
```bash
# Solution: Run optimization first
uv run optimize-prompts --model openai/gpt-4
```

**"Optimization failed"**
```bash
# Check API key
echo $OPENAI_API_KEY

# Verify training data exists
ls datasets/math_assistant/examples.json
```

**"Poor optimization scores"**
- Add more diverse training examples
- Adjust metrics in `datasets/math_assistant/metrics.py`
- Try different DSPy optimizers

### Debug Mode

```bash
# Run with verbose output
PYTHONPATH=src python -m math_assistant.optimization.optimizer
```

## Contributing

This is a prototype demonstrating the integration pattern. To extend:

1. Fork the repository
2. Add features following the established patterns
3. Test with the optimization workflow
4. Submit pull requests

## License

MIT License - see LICENSE file for details.