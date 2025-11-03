#!/usr/bin/env python3
"""
Simple CLI interface for the DeepAgents research assistant.
"""

import os
from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI

load_dotenv()

def create_research_agent():
    """Create a research agent with basic capabilities."""
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    agent = create_deep_agent(
        model=llm,
        system_prompt="You are a research assistant. Help users find information and answer questions. Use the Task tool for complex research tasks."
    )
    
    return agent

def main():
    print("🔬 Research Assistant CLI")
    print("Type 'quit' to exit")
    print("-" * 40)
    
    agent = create_research_agent()
    
    while True:
        user_input = input("\n> ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        if not user_input:
            continue
            
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
            
            # Extract the AI response from the result
            last_message = result["messages"][-1]
            print(f"\n🤖 {last_message.content}")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()