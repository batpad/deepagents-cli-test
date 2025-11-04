"""CLI for DSPy optimization"""

import click
from pathlib import Path
from .optimizer import MathOptimizer


@click.command()
@click.option(
    "--model",
    default="openai/gpt-4",
    help="Model to optimize for (e.g., openai/gpt-4, openai/gpt-3.5-turbo)"
)
@click.option(
    "--output-dir",
    help="Directory to save optimized prompts (default: project optimized_prompts/)"
)
def main(model: str, output_dir: str):
    """Optimize math assistant prompts using DSPy"""
    
    click.echo(f"🚀 Starting optimization for {model}")
    
    try:
        optimizer = MathOptimizer(model_name=model)
        
        if output_dir:
            optimizer.output_dir = Path(output_dir)
            optimizer.output_dir.mkdir(exist_ok=True)
        
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
        click.echo("Make sure the training dataset exists at datasets/math_assistant/examples.json")
        exit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}")
        exit(1)


@click.command()
@click.option(
    "--models",
    multiple=True,
    default=["openai/gpt-4", "openai/gpt-3.5-turbo"],
    help="Models to compare (can specify multiple)"
)
def compare(models):
    """Compare optimization results across multiple models"""
    
    click.echo("🔄 Comparing models...")
    results = {}
    
    for model in models:
        click.echo(f"\n📊 Optimizing {model}...")
        try:
            optimizer = MathOptimizer(model_name=model)
            result = optimizer.optimize()
            results[model] = result["test_score"] if result["success"] else 0.0
        except Exception as e:
            click.echo(f"❌ Failed for {model}: {e}")
            results[model] = 0.0
    
    # Show comparison
    click.echo("\n📈 Results:")
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    for model, score in sorted_results:
        click.echo(f"  {model}: {score:.3f}")


@click.group()
def cli():
    """DSPy optimization CLI for math assistant"""
    pass


cli.add_command(main, name="optimize")
cli.add_command(compare)


if __name__ == "__main__":
    cli()