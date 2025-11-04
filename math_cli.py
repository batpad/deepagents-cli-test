"""Interactive CLI for the math assistant"""

import click
from dotenv import load_dotenv
from agent_factory import MathAgentFactory
from math_tools import add, subtract

# Load environment variables
load_dotenv()


class MathCLI:
    """Interactive math assistant CLI"""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name
        self.factory = MathAgentFactory(model_name)
        self.agent = None
        
        # Initialize agent
        try:
            self.agent = self.factory.create_math_agent()
            self.optimization_info = self.factory.get_optimization_info()
        except Exception as e:
            self.agent = None
            self.optimization_info = {"optimized": False, "error": str(e)}
    
    def run_interactive(self):
        """Run interactive CLI"""
        self._show_header()
        
        if not self.agent:
            click.echo("❌ Could not initialize math assistant.")
            click.echo("💡 Make sure to run optimization first:")
            click.echo(f"   uv run optimize-prompts --model {self.factory.full_model_name}")
            return
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    click.echo("👋 Goodbye!")
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                elif user_input.lower() == 'info':
                    self._show_info()
                elif user_input.lower().startswith('test'):
                    self._run_test()
                elif not user_input:
                    continue
                else:
                    self._process_math_problem(user_input)
                    
            except KeyboardInterrupt:
                click.echo("\n👋 Goodbye!")
                break
            except Exception as e:
                click.echo(f"❌ Error: {e}")
    
    def _show_header(self):
        """Display CLI header"""
        click.echo("🧮 Math Assistant CLI")
        click.echo("Powered by DSPy-optimized prompts + DeepAgents")
        
        if self.optimization_info.get("optimized"):
            score = self.optimization_info.get("test_score", "unknown")
            click.echo(f"📊 Using optimized prompts (score: {score:.3f})")
        else:
            click.echo("⚠️  Using default prompts - run optimization for better performance")
        
        click.echo(f"🤖 Model: {self.model_name}")
        click.echo("Type 'help' for commands, 'quit' to exit")
        click.echo("-" * 50)
    
    def _show_help(self):
        """Show help information"""
        click.echo("""
🧮 Math Assistant Help

Commands:
  help    - Show this help message
  info    - Show optimization information
  test    - Run a quick test calculation
  quit    - Exit the application

Math Operations:
  I can help you with addition and subtraction problems.
  
Examples:
  "What is 15 + 27?"
  "Calculate 50 - 23"
  "What's 12 + 8 - 5?"
  "Compute 100 - 45 + 30"

I'll solve problems step by step and show my reasoning.
        """)
    
    def _show_info(self):
        """Show optimization information"""
        if self.optimization_info.get("optimized"):
            info = self.optimization_info
            click.echo(f"""
📊 Optimization Information:
  Model: {info['model']}
  Test Score: {info['test_score']:.3f}
  Optimized: {info['optimized_at']}
  Status: ✅ Optimized
            """)
        else:
            click.echo(f"""
📊 Optimization Information:
  Model: {self.factory.full_model_name}
  Status: ❌ Not optimized
  
💡 To optimize:
  uv run optimize-prompts --model {self.factory.full_model_name}
            """)
    
    def _run_test(self):
        """Run a quick test calculation"""
        test_problem = "What is 15 + 27?"
        click.echo(f"🧪 Testing with: {test_problem}")
        self._process_math_problem(test_problem)
    
    def _process_math_problem(self, problem: str):
        """Process a math problem with the agent"""
        click.echo(f"\n🤔 Thinking about: {problem}")
        click.echo("-" * 40)
        
        try:
            # Use the DeepAgents agent
            result = self.agent.invoke({
                "messages": [{"role": "user", "content": problem}]
            })
            
            # Extract the response
            if "messages" in result and result["messages"]:
                last_message = result["messages"][-1]
                response = last_message.content if hasattr(last_message, 'content') else str(last_message)
                
                click.echo(f"🧮 {response}")
            else:
                click.echo("❌ No response from agent")
                
        except Exception as e:
            click.echo(f"❌ Error processing problem: {e}")
            
            # Fallback: try direct calculation if it's simple
            try:
                self._fallback_calculation(problem)
            except:
                pass
    
    def _fallback_calculation(self, problem: str):
        """Simple fallback for basic calculations"""
        # Very basic parsing for simple cases
        problem = problem.lower().replace("what is", "").replace("calculate", "").replace("?", "").strip()
        
        if "+" in problem and "-" not in problem:
            parts = problem.split("+")
            if len(parts) == 2:
                try:
                    a, b = float(parts[0].strip()), float(parts[1].strip())
                    result = add(a, b)
                    click.echo(f"🔧 Fallback calculation: {a} + {b} = {result}")
                except:
                    pass
        elif "-" in problem and "+" not in problem:
            parts = problem.split("-")
            if len(parts) == 2:
                try:
                    a, b = float(parts[0].strip()), float(parts[1].strip())
                    result = subtract(a, b)
                    click.echo(f"🔧 Fallback calculation: {a} - {b} = {result}")
                except:
                    pass


@click.command()
@click.option(
    "--model",
    default="gpt-4",
    help="Model to use (e.g., gpt-4, gpt-3.5-turbo)"
)
@click.option(
    "--problem",
    help="Single problem to solve (non-interactive mode)"
)
def main(model: str, problem: str):
    """Math Assistant CLI - powered by DSPy + DeepAgents"""
    
    cli = MathCLI(model)
    
    if problem:
        # Non-interactive mode
        cli._process_math_problem(problem)
    else:
        # Interactive mode
        cli.run_interactive()


if __name__ == "__main__":
    main()