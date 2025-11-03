#!/usr/bin/env python3
"""
Simple DeepAgents demo with one subagent.
"""

import os
from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI

load_dotenv()

def main():
    print("Creating simple agent with subagent...")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    # Create main agent that can use the built-in general-purpose subagent
    agent = create_deep_agent(
        model=llm,
        system_prompt="You are a helpful assistant. Use the Task tool when you need to delegate complex work."
    )
    
    print("Agent created!")
    
    # Test it
    result = agent.invoke({"messages": [{"role": "user", "content": "What is 15 + 27?"}]})
    print(f"Result: {result}")

if __name__ == "__main__":
    main()