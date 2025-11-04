"""DeepAgents integration with optimized prompts"""

import json
from pathlib import Path
from typing import Optional
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from math_tools import add, subtract, get_tool_description, get_langchain_tools


class PromptManager:
    """Manage optimized prompts from DSPy"""
    
    def __init__(self, prompts_dir: Optional[Path] = None):
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent / "optimized_prompts"
        
        self.prompts_dir = Path(prompts_dir)
        self.current_model = None
        self.prompt_cache = {}
    
    def load_prompt_for_model(self, model_name: str) -> str:
        """Load optimized prompt for specific model"""
        model_key = model_name.replace("/", "_").replace(":", "_")
        prompt_file = self.prompts_dir / f"math_assistant_{model_key}.json"
        
        if not prompt_file.exists():
            raise FileNotFoundError(
                f"No optimized prompt found for {model_name}. "
                f"Run: uv run optimize-prompts --model {model_name}"
            )
        
        with open(prompt_file) as f:
            data = json.load(f)
        
        self.current_model = model_name
        self.prompt_cache[model_name] = data["prompt"]
        return data["prompt"]
    
    def get_system_prompt(self, model_name: str) -> str:
        """Get formatted system prompt for DeepAgents"""
        if model_name not in self.prompt_cache:
            self.load_prompt_for_model(model_name)
        
        base_prompt = self.prompt_cache[model_name]
        
        # Add DeepAgents-specific tool information
        deepagents_addition = f"""

{get_tool_description()}

You can also use these DeepAgents tools if needed:
- write_file: Save calculations or work to files
- read_file: Read previous work from files
- write_todos: Track multi-step problems

For math problems, focus on using the math tools (add, subtract) and providing clear reasoning."""
        
        return base_prompt + deepagents_addition


class MathAgentFactory:
    """Create DeepAgents math assistants with optimized prompts"""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name
        self.full_model_name = f"openai/{model_name}" if not model_name.startswith("openai/") else model_name
        self.llm = ChatOpenAI(model=model_name)
        self.prompt_manager = PromptManager()
    
    def create_math_agent(self):
        """Create a math assistant agent with optimized prompts"""
        try:
            # Load optimized prompt
            system_prompt = self.prompt_manager.get_system_prompt(self.full_model_name)
            
            # Create the agent with DeepAgents and math tools
            agent = create_deep_agent(
                model=self.llm,
                system_prompt=system_prompt,
                tools=get_langchain_tools()  # Pass the LangChain tools
            )
            
            return agent
            
        except FileNotFoundError as e:
            print(f"❌ {e}")
            print(f"💡 Run: uv run optimize-prompts --model {self.full_model_name}")
            raise
    
    def check_optimization_status(self) -> bool:
        """Check if optimization exists for current model"""
        try:
            self.prompt_manager.load_prompt_for_model(self.full_model_name)
            return True
        except FileNotFoundError:
            return False
    
    def get_optimization_info(self) -> dict:
        """Get information about the current optimization"""
        if not self.check_optimization_status():
            return {"optimized": False}
        
        model_key = self.full_model_name.replace("/", "_").replace(":", "_")
        prompt_file = self.prompt_manager.prompts_dir / f"math_assistant_{model_key}.json"
        
        with open(prompt_file) as f:
            data = json.load(f)
        
        return {
            "optimized": True,
            "model": data["model"],
            "test_score": data["test_score"],
            "optimized_at": data["optimized_at"]
        }


# For backwards compatibility and easy imports
def create_math_assistant(model_name: str = "gpt-4"):
    """Convenience function to create a math assistant"""
    factory = MathAgentFactory(model_name)
    return factory.create_math_agent()


if __name__ == "__main__":
    # Simple test
    try:
        factory = MathAgentFactory("gpt-4")
        info = factory.get_optimization_info()
        print(f"Optimization info: {info}")
        
        if info["optimized"]:
            agent = factory.create_math_agent()
            print("✅ Math agent created successfully!")
        else:
            print("❌ No optimization found. Run optimization first.")
            
    except Exception as e:
        print(f"Error: {e}")