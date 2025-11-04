"""DSPy optimization pipeline for math assistant"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Any
import dspy
from math_tools import add, subtract


class MathReasoning(dspy.Signature):
    """Solve math problems step by step using available tools"""
    problem = dspy.InputField(desc="math problem to solve")
    reasoning = dspy.OutputField(desc="step-by-step reasoning explaining how to solve the problem")
    final_answer = dspy.OutputField(desc="final answer with clear explanation")


class MathAssistant(dspy.Module):
    """DSPy module for math problem solving"""
    
    def __init__(self):
        self.solve = dspy.ChainOfThought(MathReasoning)
    
    def forward(self, problem: str):
        """Solve a math problem"""
        result = self.solve(problem=problem)
        return result


class MathOptimizer:
    """Optimize math assistant prompts using DSPy"""
    
    def __init__(self, model_name: str = "openai/gpt-4"):
        self.model_name = model_name
        self.lm = dspy.LM(model_name)
        dspy.configure(lm=self.lm)
        
        self.assistant = MathAssistant()
        self.dataset_path = Path(__file__).parent / "datasets" / "examples.json"
        self.output_dir = Path(__file__).parent / "optimized_prompts"
        self.output_dir.mkdir(exist_ok=True)
    
    def load_training_data(self) -> List[dspy.Example]:
        """Load and convert training examples"""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Training data not found: {self.dataset_path}")
        
        with open(self.dataset_path) as f:
            raw_data = json.load(f)
        
        examples = []
        for item in raw_data:
            example = dspy.Example(
                problem=item["problem"],
                reasoning=item["reasoning"],
                final_answer=item["final_answer"]
            ).with_inputs("problem")
            examples.append(example)
        
        return examples
    
    def math_accuracy_metric(self, prediction, example):
        """Evaluate prediction quality"""
        # Import metrics from datasets directory
        import sys
        sys.path.append(str(Path(__file__).parent / "datasets"))
        from metrics import math_accuracy_metric
        
        return math_accuracy_metric(prediction, example)
    
    def optimize(self) -> Dict[str, Any]:
        """Run optimization process"""
        print(f"🔧 Optimizing math assistant for {self.model_name}")
        
        # Load training data
        examples = self.load_training_data()
        print(f"📊 Loaded {len(examples)} training examples")
        
        # Split into train/test
        train_size = int(len(examples) * 0.8)
        trainset = examples[:train_size]
        testset = examples[train_size:]
        
        print(f"🎯 Training on {len(trainset)} examples, testing on {len(testset)}")
        
        # Configure optimizer - use BootstrapFewShot for simpler setup
        optimizer = dspy.BootstrapFewShot(
            metric=self.math_accuracy_metric,
            max_bootstrapped_demos=4,
            max_labeled_demos=4
        )
        
        # Optimize
        print("⚡ Starting optimization...")
        try:
            optimized_assistant = optimizer.compile(
                self.assistant,
                trainset=trainset
            )
            
            # Test performance
            print("📈 Testing optimized assistant...")
            test_scores = []
            for example in testset:
                try:
                    prediction = optimized_assistant(problem=example.problem)
                    score = self.math_accuracy_metric(prediction, example)
                    test_scores.append(score)
                except Exception as e:
                    print(f"❌ Test failed for '{example.problem}': {e}")
                    test_scores.append(0.0)
            
            avg_score = sum(test_scores) / len(test_scores) if test_scores else 0.0
            print(f"🎯 Average test score: {avg_score:.3f}")
            
            # Extract and save optimized prompt
            optimized_prompt = self._extract_prompt(optimized_assistant)
            self._save_optimized_prompt(optimized_prompt, avg_score)
            
            return {
                "model": self.model_name,
                "test_score": avg_score,
                "prompt": optimized_prompt,
                "success": True
            }
            
        except Exception as e:
            print(f"❌ Optimization failed: {e}")
            return {
                "model": self.model_name,
                "error": str(e),
                "success": False
            }
    
    def _extract_prompt(self, optimized_assistant) -> str:
        """Extract optimized prompt from DSPy module"""
        try:
            # Try to get the signature and any demonstrations
            signature = optimized_assistant.solve.signature
            
            # Build the prompt
            prompt_parts = [
                "You are a helpful math assistant that solves problems step by step.",
                "",
                f"Task: {signature.__doc__}",
                "",
                "Available tools:",
                "- add(a, b): Add two numbers",
                "- subtract(a, b): Subtract b from a",
                "",
                "Instructions:",
                "1. Break down the problem step by step",
                "2. Explain your reasoning clearly", 
                "3. Use the appropriate tools for calculations",
                "4. Provide a clear final answer",
                "",
            ]
            
            # Add any demonstrations/examples if available
            if hasattr(optimized_assistant.solve, 'demos') and optimized_assistant.solve.demos:
                prompt_parts.append("Examples:")
                for demo in optimized_assistant.solve.demos[:3]:  # Limit to 3 examples
                    prompt_parts.append(f"Problem: {demo.problem}")
                    prompt_parts.append(f"Reasoning: {demo.reasoning}")
                    prompt_parts.append(f"Answer: {demo.final_answer}")
                    prompt_parts.append("")
            
            prompt_parts.extend([
                "Now solve this problem:",
                "Problem: {problem}",
                "",
                "Provide your reasoning and final answer."
            ])
            
            return "\n".join(prompt_parts)
            
        except Exception as e:
            print(f"⚠️ Could not extract optimized prompt, using default: {e}")
            return self._get_default_prompt()
    
    def _get_default_prompt(self) -> str:
        """Fallback prompt if extraction fails"""
        return """You are a helpful math assistant that solves problems step by step.

Task: Solve math problems step by step using available tools

Available tools:
- add(a, b): Add two numbers
- subtract(a, b): Subtract b from a

Instructions:
1. Break down the problem step by step
2. Explain your reasoning clearly
3. Use the appropriate tools for calculations
4. Provide a clear final answer

Now solve this problem:
Problem: {problem}

Provide your reasoning and final answer."""
    
    def _save_optimized_prompt(self, prompt: str, score: float):
        """Save optimized prompt to file"""
        model_key = self.model_name.replace("/", "_").replace(":", "_")
        output_file = self.output_dir / f"math_assistant_{model_key}.json"
        
        prompt_data = {
            "agent_type": "math_assistant",
            "model": self.model_name,
            "prompt": prompt,
            "test_score": score,
            "optimized_at": __import__('datetime').datetime.now().isoformat()
        }
        
        with open(output_file, 'w') as f:
            json.dump(prompt_data, f, indent=2)
        
        print(f"💾 Optimized prompt saved to {output_file}")
        return output_file
    
    def load_optimized_prompt(self) -> str:
        """Load optimized prompt for current model"""
        model_key = self.model_name.replace("/", "_").replace(":", "_")
        prompt_file = self.output_dir / f"math_assistant_{model_key}.json"
        
        if not prompt_file.exists():
            raise FileNotFoundError(f"No optimized prompt found for {self.model_name}")
        
        with open(prompt_file) as f:
            data = json.load(f)
        
        return data["prompt"]


import click

@click.command()
@click.option(
    "--model",
    default="openai/gpt-4",
    help="Model to optimize for (e.g., openai/gpt-4, openai/gpt-3.5-turbo)"
)
def main(model: str):
    """Optimize math assistant prompts using DSPy"""
    
    click.echo(f"🚀 Starting optimization for {model}")
    
    try:
        optimizer = MathOptimizer(model_name=model)
        result = optimizer.optimize()
        
        if result["success"]:
            click.echo(f"✅ Optimization completed successfully!")
            click.echo(f"📊 Test score: {result['test_score']:.3f}")
            click.echo(f"💾 Prompt saved for model: {model}")
        else:
            click.echo(f"❌ Optimization failed: {result['error']}")
            exit(1)
            
    except FileNotFoundError as e:
        click.echo(f"❌ Error: {e}")
        click.echo("Make sure the training dataset exists at datasets/examples.json")
        exit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}")
        exit(1)

if __name__ == "__main__":
    main()