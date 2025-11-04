"""Simple math tools for the assistant"""

from typing import Union


def add(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """
    Add two numbers together.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        The sum of a and b
    """
    return a + b


def subtract(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
    """
    Subtract b from a.
    
    Args:
        a: Number to subtract from
        b: Number to subtract
        
    Returns:
        The difference a - b
    """
    return a - b


# Tool registry for easy access
MATH_TOOLS = {
    "add": add,
    "subtract": subtract
}


def get_tool_description():
    """Get description of available math tools"""
    return """Available math tools:
- add(a, b): Add two numbers together
- subtract(a, b): Subtract b from a

Use these tools to perform calculations step by step."""


def execute_tool(tool_name: str, *args) -> Union[int, float]:
    """Execute a math tool by name"""
    if tool_name not in MATH_TOOLS:
        raise ValueError(f"Unknown tool: {tool_name}. Available tools: {list(MATH_TOOLS.keys())}")
    
    return MATH_TOOLS[tool_name](*args)


if __name__ == "__main__":
    # Simple test
    print("Testing math tools:")
    print(f"add(15, 27) = {add(15, 27)}")
    print(f"subtract(50, 23) = {subtract(50, 23)}")
    print(f"execute_tool('add', 12, 8) = {execute_tool('add', 12, 8)}")