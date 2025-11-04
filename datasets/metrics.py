"""Evaluation metrics for math assistant optimization"""

import re
from typing import Any, Dict, List


def extract_numbers_from_text(text: str) -> List[float]:
    """Extract numeric values from text"""
    # Find all numbers (including decimals)
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    return [float(n) for n in numbers]


def extract_final_answer(text: str) -> float:
    """Extract the final numeric answer from response text"""
    # Look for patterns like "= 42" or "answer is 42"
    patterns = [
        r'=\s*(-?\d+(?:\.\d+)?)',
        r'(?:answer|result)(?:\s+is)?\s*(-?\d+(?:\.\d+)?)',
        r'(-?\d+(?:\.\d+)?)\s*$'  # Number at end of text
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return float(match.group(1))
    
    # Fallback: get the last number in the text
    numbers = extract_numbers_from_text(text)
    if numbers:
        return numbers[-1]
    
    return None


def check_tool_usage(prediction: Any, expected_tools: List[Dict]) -> float:
    """Check if the correct tools were mentioned in the reasoning"""
    pred_text = str(prediction).lower()
    
    # Check if correct tools are mentioned
    tool_mentions = {
        'add': 'add' in pred_text or 'addition' in pred_text or '+' in pred_text,
        'subtract': 'subtract' in pred_text or 'subtraction' in pred_text or '-' in pred_text
    }
    
    expected_tool_types = set()
    for tool_call in expected_tools:
        expected_tool_types.add(tool_call['tool'])
    
    # Calculate tool mention accuracy
    correct_mentions = 0
    for tool in expected_tool_types:
        if tool_mentions.get(tool, False):
            correct_mentions += 1
    
    if not expected_tool_types:
        return 1.0
    
    return correct_mentions / len(expected_tool_types)


def math_accuracy_metric(prediction: Any, example: Any) -> float:
    """
    Evaluate math assistant accuracy
    
    Checks:
    1. Correct final numeric answer
    2. Appropriate tool usage mentioned
    3. Logical reasoning structure
    """
    # Extract expected answer from the example
    expected_answer_text = example.final_answer
    expected_number = extract_final_answer(expected_answer_text)
    
    # Extract predicted answer
    prediction_text = str(prediction)
    predicted_number = extract_final_answer(prediction_text)
    
    # Check numeric accuracy
    numeric_score = 0.0
    if expected_number is not None and predicted_number is not None:
        if abs(expected_number - predicted_number) < 0.001:  # Allow for small floating point errors
            numeric_score = 1.0
    
    # Check tool usage
    tool_score = check_tool_usage(prediction, example.tool_calls)
    
    # Check if reasoning is present
    reasoning_score = 0.0
    if len(prediction_text) > 20:  # Has some explanation
        reasoning_score = 0.5
    if any(word in prediction_text.lower() for word in ['first', 'then', 'step', 'calculate']):
        reasoning_score = 1.0
    
    # Weighted average
    final_score = (
        numeric_score * 0.6 +      # 60% for correct answer
        tool_score * 0.3 +         # 30% for tool usage
        reasoning_score * 0.1      # 10% for reasoning quality
    )
    
    return final_score


def simple_accuracy_metric(prediction: Any, example: Any) -> float:
    """Simplified metric that just checks if the final number is correct"""
    expected_number = extract_final_answer(example.final_answer)
    predicted_number = extract_final_answer(str(prediction))
    
    if expected_number is not None and predicted_number is not None:
        return 1.0 if abs(expected_number - predicted_number) < 0.001 else 0.0
    
    return 0.0