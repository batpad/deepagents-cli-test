# DSPy-Only Integration Plan

## Overview

This plan eliminates the complexity of integrating LangGraph + DSPy by using **only DSPy** for the entire geospatial workflow. Instead of fighting two frameworks, we leverage DSPy's natural composition patterns and end-to-end optimization capabilities.

## Critiques of LangGraph + DSPy Integration

### 1. **Framework Conflict**
- **Problem**: LangGraph and DSPy both want to control workflow orchestration
- **Specific Issue**: LangGraph manages state/routing while DSPy wants to optimize complete pipelines
- **Result**: Fighting two opinionated frameworks instead of using one properly

### 2. **Fragmented Optimization**
- **Problem**: DSPy optimization gets broken into isolated LangGraph node pieces
- **Specific Issue**: DSPy's strength is end-to-end pipeline optimization, not isolated components
- **Result**: Suboptimal results because DSPy can't see the full workflow context

### 3. **Multiple State Systems**
- **Problem**: LangGraph state + DSPy internal state + DeepAgents state
- **Specific Issue**: Complex serialization, checkpointing, debugging across boundaries
- **Result**: Added complexity without clear benefits

### 4. **Manual vs Learned Routing**
- **Problem**: LangGraph uses manual conditional routing, DSPy can learn optimal routing
- **Specific Issue**: Hard-coding routing logic instead of letting DSPy discover optimal paths
- **Result**: Missing opportunity for learned workflow optimization

### 5. **Tool vs Reasoning Mismatch**
- **Problem**: Geospatial workflows are tool-heavy (scrape/process/visualize) vs reasoning-heavy
- **Specific Issue**: DSPy excels at reasoning chains, we're using it for tool orchestration
- **Result**: Using DSPy outside its sweet spot

### 6. **Cognitive Overhead**
- **Problem**: Three frameworks (LangGraph + DSPy + DeepAgents) to learn and maintain
- **Specific Issue**: Each has different patterns, error handling, debugging approaches
- **Result**: High development and maintenance burden

## DSPy-Only Architecture

### Core Philosophy
- **Single Framework**: DSPy handles everything from routing to execution
- **End-to-End Optimization**: Optimize complete workflows, not fragments
- **Learned Patterns**: Let DSPy discover optimal workflows through training
- **Natural Composition**: Use DSPy's built-in module composition

### High-Level Design

```
User Request → DSPy Pipeline → Optimized Workflow → Results
                     ↓
            [Task Analysis] → [Data Collection] → [Processing] → [Analysis] → [Visualization]
                     ↓              ↓              ↓           ↓              ↓
                 DSPy Module   DSPy Module    DSPy Module  DSPy Module   DSPy Module
```

## Core Implementation

### 1. Complete Geospatial Pipeline

```python
# core/geospatial_pipeline.py
import dspy
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List

class TaskAnalysis(dspy.Signature):
    """Analyze user request and determine workflow requirements"""
    user_request = dspy.InputField(desc="user's geospatial task request")
    task_breakdown = dspy.OutputField(desc="structured breakdown of required steps")
    data_requirements = dspy.OutputField(desc="what data is needed and from where")
    analysis_type = dspy.OutputField(desc="type of analysis required")
    output_format = dspy.OutputField(desc="desired output format")

class DataCollection(dspy.Signature):
    """Generate strategy and code for collecting geospatial data"""
    requirements = dspy.InputField(desc="data collection requirements")
    source_info = dspy.InputField(desc="information about data sources")
    collection_strategy = dspy.OutputField(desc="approach for collecting the data")
    python_code = dspy.OutputField(desc="executable Python code for data collection")

class DataProcessing(dspy.Signature):
    """Generate code for processing and cleaning geospatial data"""
    raw_data_description = dspy.InputField(desc="description of raw data format and content")
    target_format = dspy.InputField(desc="desired output format")
    processing_steps = dspy.OutputField(desc="step-by-step processing approach")
    python_code = dspy.OutputField(desc="executable Python code for data processing")

class DataAnalysis(dspy.Signature):
    """Analyze processed geospatial data to extract insights"""
    data_description = dspy.InputField(desc="description of processed data")
    analysis_questions = dspy.InputField(desc="specific questions to answer")
    patterns = dspy.OutputField(desc="identified patterns and trends")
    insights = dspy.OutputField(desc="actionable insights and findings")
    statistical_summary = dspy.OutputField(desc="key statistics and metrics")

class VisualizationGeneration(dspy.Signature):
    """Generate visualization code for geospatial analysis"""
    analysis_results = dspy.InputField(desc="results from data analysis")
    viz_requirements = dspy.InputField(desc="visualization requirements")
    chart_strategy = dspy.OutputField(desc="visualization approach and chart types")
    python_code = dspy.OutputField(desc="executable Python code for creating visualizations")

class GeospatialPipeline(dspy.Module):
    """Complete geospatial data processing pipeline using only DSPy"""
    
    def __init__(self):
        # Core workflow modules
        self.task_analyzer = dspy.ChainOfThought(TaskAnalysis)
        self.data_collector = dspy.ProgramOfThought(DataCollection)
        self.data_processor = dspy.ProgramOfThought(DataProcessing)
        self.data_analyzer = dspy.ChainOfThought(DataAnalysis)
        self.visualizer = dspy.ProgramOfThought(VisualizationGeneration)
        
        # Workflow state
        self.current_workflow = {}
        self.results = {}
        self.files_created = []
    
    def forward(self, user_request: str) -> Dict[str, Any]:
        """Execute the complete geospatial workflow"""
        
        # Step 1: Analyze the user's request
        print("🔍 Analyzing task requirements...")
        analysis = self.task_analyzer(user_request=user_request)
        self.current_workflow["analysis"] = analysis
        
        # Step 2: Collect data if needed
        if "collect" in analysis.task_breakdown.lower() or "scrape" in analysis.task_breakdown.lower():
            print("📥 Collecting data...")
            collection = self.data_collector(
                requirements=analysis.data_requirements,
                source_info=analysis.task_breakdown
            )
            self.current_workflow["collection"] = collection
            
            # Execute data collection code
            collected_data = self._execute_code(collection.python_code, "data_collection")
            self.results["raw_data"] = collected_data
        
        # Step 3: Process data if needed
        if "process" in analysis.task_breakdown.lower() or "clean" in analysis.task_breakdown.lower():
            print("⚙️ Processing data...")
            processing = self.data_processor(
                raw_data_description=str(self.results.get("raw_data", "No raw data")),
                target_format=analysis.output_format
            )
            self.current_workflow["processing"] = processing
            
            # Execute processing code
            processed_data = self._execute_code(processing.python_code, "data_processing")
            self.results["processed_data"] = processed_data
        
        # Step 4: Analyze data
        if "analyz" in analysis.task_breakdown.lower() or "pattern" in analysis.task_breakdown.lower():
            print("📊 Analyzing data...")
            data_description = str(self.results.get("processed_data", self.results.get("raw_data", "")))
            analysis_result = self.data_analyzer(
                data_description=data_description,
                analysis_questions=analysis.analysis_type
            )
            self.current_workflow["analysis"] = analysis_result
            self.results["analysis"] = analysis_result
        
        # Step 5: Create visualizations if needed
        if "visual" in analysis.task_breakdown.lower() or "chart" in analysis.task_breakdown.lower():
            print("📈 Creating visualizations...")
            visualization = self.visualizer(
                analysis_results=str(self.results.get("analysis", "")),
                viz_requirements=analysis.output_format
            )
            self.current_workflow["visualization"] = visualization
            
            # Execute visualization code
            viz_result = self._execute_code(visualization.python_code, "visualization")
            self.results["visualization"] = viz_result
        
        # Compile final results
        return self._compile_results()
    
    def _execute_code(self, code: str, step_name: str) -> Any:
        """Safely execute generated Python code"""
        try:
            # Create execution environment with necessary imports
            exec_globals = {
                'requests': requests,
                'pd': pd,
                'plt': plt,
                'json': json,
                'np': __import__('numpy'),
                '__builtins__': __builtins__
            }
            
            # Execute the code
            exec(code, exec_globals)
            
            # Try to get result variable if it exists
            result = exec_globals.get('result', f"Code executed successfully for {step_name}")
            
            # Save code to file for reference
            filename = f"{step_name}_code.py"
            with open(filename, 'w') as f:
                f.write(code)
            self.files_created.append(filename)
            
            return result
            
        except Exception as e:
            return f"Error in {step_name}: {str(e)}"
    
    def _compile_results(self) -> Dict[str, Any]:
        """Compile all workflow results into final output"""
        return {
            "workflow": self.current_workflow,
            "results": self.results,
            "files_created": self.files_created,
            "summary": self._generate_summary()
        }
    
    def _generate_summary(self) -> str:
        """Generate a summary of the workflow execution"""
        summary_parts = ["🌍 Geospatial Workflow Complete!"]
        
        if "collection" in self.current_workflow:
            summary_parts.append("✅ Data Collection: " + self.current_workflow["collection"].collection_strategy[:100] + "...")
        
        if "processing" in self.current_workflow:
            summary_parts.append("✅ Data Processing: " + self.current_workflow["processing"].processing_steps[:100] + "...")
        
        if "analysis" in self.results:
            summary_parts.append("✅ Analysis: " + self.results["analysis"].insights[:100] + "...")
        
        if "visualization" in self.current_workflow:
            summary_parts.append("✅ Visualization: " + self.current_workflow["visualization"].chart_strategy[:100] + "...")
        
        summary_parts.append(f"📁 Files Created: {', '.join(self.files_created)}")
        
        return "\n".join(summary_parts)
```

### 2. Simple CLI Interface

```python
# cli/simple_cli.py
#!/usr/bin/env python3
"""
Simple CLI for DSPy-only geospatial workflows
"""

import dspy
import pickle
import os
from pathlib import Path
from core.geospatial_pipeline import GeospatialPipeline

class GeospatialCLI:
    def __init__(self, model_name: str = "openai/gpt-4"):
        # Configure DSPy
        self.lm = dspy.LM(model_name)
        dspy.configure(lm=self.lm)
        
        # Initialize pipeline
        self.pipeline = GeospatialPipeline()
        
        # Load optimized version if available
        self.optimized_path = Path("models/optimized_pipeline.pkl")
        if self.optimized_path.exists():
            print("📈 Loading optimized pipeline...")
            with open(self.optimized_path, 'rb') as f:
                self.pipeline = pickle.load(f)
        
        # Create workspace
        self.workspace = Path("workspace")
        self.workspace.mkdir(exist_ok=True)
        os.chdir(self.workspace)
    
    def run_interactive(self):
        """Interactive CLI mode"""
        print("🌍 DSPy Geospatial Processing CLI")
        print("Pure DSPy implementation - no LangGraph")
        print("Commands: 'help', 'optimize', 'status', 'quit'")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'help':
                    self.show_help()
                elif user_input.lower() == 'optimize':
                    self.run_optimization()
                elif user_input.lower() == 'status':
                    self.show_status()
                else:
                    self.process_request(user_input)
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def process_request(self, request: str):
        """Process a user request through the DSPy pipeline"""
        print(f"\n🚀 Processing: {request}")
        print("-" * 40)
        
        try:
            # Run the complete pipeline
            result = self.pipeline(user_request=request)
            
            # Display results
            print("\n" + result["summary"])
            
            if result["results"].get("analysis"):
                print(f"\n📊 Key Insights:")
                print(result["results"]["analysis"].insights)
            
            if result["files_created"]:
                print(f"\n📁 Files created in workspace:")
                for file in result["files_created"]:
                    print(f"  - {file}")
                    
        except Exception as e:
            print(f"❌ Error processing request: {e}")
    
    def run_optimization(self):
        """Optimize the pipeline using training data"""
        print("🔧 Starting pipeline optimization...")
        
        # Check if training data exists
        training_path = Path("../datasets/complete_workflows.json")
        if not training_path.exists():
            print("❌ No training data found. Create datasets/complete_workflows.json first.")
            return
        
        try:
            # Load training examples
            import json
            with open(training_path) as f:
                training_data = json.load(f)
            
            # Convert to DSPy examples
            examples = []
            for item in training_data:
                example = dspy.Example(
                    user_request=item["user_request"],
                    expected_output=item["expected_output"]
                ).with_inputs("user_request")
                examples.append(example)
            
            # Define metric
            def workflow_quality(prediction, example):
                # Simple metric - could be more sophisticated
                pred_text = str(prediction)
                expected_text = str(example.expected_output)
                
                # Check for key components
                has_analysis = "analysis" in pred_text.lower()
                has_data = "data" in pred_text.lower()
                has_insights = "insight" in pred_text.lower()
                
                return int(has_analysis and has_data and has_insights)
            
            # Optimize
            optimizer = dspy.MIPROv2(metric=workflow_quality)
            optimized_pipeline = optimizer.compile(
                self.pipeline,
                trainset=examples[:int(len(examples) * 0.8)],
                num_trials=20
            )
            
            # Save optimized pipeline
            self.optimized_path.parent.mkdir(exist_ok=True)
            with open(self.optimized_path, 'wb') as f:
                pickle.dump(optimized_pipeline, f)
            
            # Use optimized version
            self.pipeline = optimized_pipeline
            print("✅ Optimization complete! Pipeline saved and loaded.")
            
        except Exception as e:
            print(f"❌ Optimization failed: {e}")
    
    def show_help(self):
        """Show help information"""
        print("""
🌍 DSPy Geospatial CLI Help

Commands:
  help      - Show this help message
  optimize  - Optimize pipeline with training data
  status    - Show system status
  quit      - Exit application

Example Requests:
  "Analyze earthquake patterns in California from USGS data"
  "Collect and visualize population data for major cities"
  "Process wildfire data and identify high-risk areas"
  "Download climate data and create trend analysis"

The system will automatically:
  - Determine what data to collect
  - Generate code for collection and processing
  - Perform analysis and extract insights
  - Create visualizations as needed
  - Save all code and results to workspace
        """)
    
    def show_status(self):
        """Show system status"""
        optimized = "✅ Optimized" if self.optimized_path.exists() else "❌ Not optimized"
        
        print(f"""
📊 System Status:
  Model: {self.lm}
  Pipeline: {optimized}
  Workspace: {self.workspace}
  
🧠 DSPy Modules:
  ✓ Task Analysis (ChainOfThought)
  ✓ Data Collection (ProgramOfThought)  
  ✓ Data Processing (ProgramOfThought)
  ✓ Data Analysis (ChainOfThought)
  ✓ Visualization (ProgramOfThought)
        """)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="DSPy Geospatial CLI")
    parser.add_argument("--model", default="openai/gpt-4", help="LLM model to use")
    parser.add_argument("--request", help="Single request to process")
    
    args = parser.parse_args()
    
    cli = GeospatialCLI(args.model)
    
    if args.request:
        cli.process_request(args.request)
    else:
        cli.run_interactive()

if __name__ == "__main__":
    main()
```

### 3. File Management Through DSPy

```python
# core/file_manager.py
import dspy
import os
import json
import pickle
from pathlib import Path

class FileOperation(dspy.Signature):
    """Plan file operations for geospatial workflows"""
    operation_type = dspy.InputField(desc="type of file operation needed")
    data_description = dspy.InputField(desc="description of data to handle")
    file_strategy = dspy.OutputField(desc="strategy for organizing and managing files")
    code = dspy.OutputField(desc="Python code to execute the file operations")

class FileManager(dspy.Module):
    """Handle all file operations through DSPy"""
    
    def __init__(self):
        self.file_planner = dspy.ChainOfThought(FileOperation)
        self.workspace = Path("workspace")
        self.workspace.mkdir(exist_ok=True)
    
    def forward(self, operation_type: str, data_description: str) -> dict:
        """Execute file operations based on DSPy planning"""
        
        # Plan the file operation
        plan = self.file_planner(
            operation_type=operation_type,
            data_description=data_description
        )
        
        # Execute the generated code
        try:
            exec_globals = {
                'os': os,
                'json': json,
                'pickle': pickle,
                'Path': Path,
                'workspace': self.workspace
            }
            
            exec(plan.code, exec_globals)
            result = exec_globals.get('result', 'File operation completed')
            
            return {
                "strategy": plan.file_strategy,
                "code": plan.code,
                "result": result,
                "success": True
            }
            
        except Exception as e:
            return {
                "strategy": plan.file_strategy,
                "code": plan.code,
                "error": str(e),
                "success": False
            }

# Example usage patterns
class WorkflowFileManager:
    """Specific file management for geospatial workflows"""
    
    def __init__(self):
        self.file_manager = FileManager()
    
    def save_data(self, data, data_type: str):
        """Save data with appropriate format and naming"""
        return self.file_manager(
            operation_type=f"save_{data_type}_data",
            data_description=f"{data_type} data: {str(data)[:200]}..."
        )
    
    def load_data(self, data_type: str, source_info: str):
        """Load data based on type and source information"""
        return self.file_manager(
            operation_type=f"load_{data_type}_data",
            data_description=f"Load {data_type} from {source_info}"
        )
    
    def organize_results(self, workflow_results: dict):
        """Organize all workflow results into structured files"""
        return self.file_manager(
            operation_type="organize_workflow_results",
            data_description=f"Workflow results: {str(workflow_results)[:300]}..."
        )
```

### 4. Training Data Structure

```json
// datasets/complete_workflows.json
[
  {
    "user_request": "Analyze recent earthquake activity in California",
    "expected_output": {
      "data_collected": "USGS earthquake data for California, last 30 days",
      "processing": "Filtered by magnitude > 3.0, geocoded locations",
      "analysis": "56 earthquakes found, clustering along San Andreas fault, magnitude range 3.1-5.8",
      "insights": "Increased activity near Los Angeles, deeper events correlate with higher magnitude",
      "visualization": "Map showing earthquake locations sized by magnitude, timeline plot"
    }
  },
  {
    "user_request": "Download and visualize population growth in Texas cities",
    "expected_output": {
      "data_collected": "US Census population data for Texas metropolitan areas 2010-2020",
      "processing": "Cleaned city names, calculated growth rates, normalized populations",
      "analysis": "Houston +15.8%, Austin +22.4%, Dallas +18.7% growth over decade",
      "insights": "Austin leads growth rate, Houston leads absolute growth, correlation with tech employment",
      "visualization": "Bar chart of growth rates, map of city sizes, trend lines over time"
    }
  },
  {
    "user_request": "Process wildfire data and identify high-risk areas",
    "expected_output": {
      "data_collected": "MODIS fire detection data, weather patterns, vegetation indices",
      "processing": "Integrated multi-source data, calculated fire risk metrics, spatial clustering",
      "analysis": "High-risk zones identified in foothill regions, correlation with drought conditions",
      "insights": "Risk increases 340% during Santa Ana wind events, urban-wildland interface most vulnerable",
      "visualization": "Risk heat map, time series of fire activity, correlation plots with weather"
    }
  }
]
```

### 5. Optimization Workflow

```python
# optimization/optimizer.py
import dspy
import json
from pathlib import Path
from core.geospatial_pipeline import GeospatialPipeline

class PipelineOptimizer:
    """Optimize the complete DSPy geospatial pipeline"""
    
    def __init__(self, model_name: str = "openai/gpt-4"):
        self.model_name = model_name
        dspy.configure(lm=dspy.LM(model_name))
        
        self.pipeline = GeospatialPipeline()
        self.training_data_path = Path("datasets/complete_workflows.json")
        self.output_path = Path("models/optimized_pipeline.pkl")
    
    def load_training_data(self):
        """Load and prepare training examples"""
        if not self.training_data_path.exists():
            raise FileNotFoundError(f"Training data not found: {self.training_data_path}")
        
        with open(self.training_data_path) as f:
            raw_data = json.load(f)
        
        examples = []
        for item in raw_data:
            example = dspy.Example(
                user_request=item["user_request"],
                expected_output=item["expected_output"]
            ).with_inputs("user_request")
            examples.append(example)
        
        return examples
    
    def workflow_quality_metric(self, prediction, example):
        """Evaluate workflow quality"""
        pred_str = str(prediction).lower()
        expected = example.expected_output
        
        # Check for required components
        components = {
            "data_collection": any(term in pred_str for term in ["collect", "download", "fetch", "scrape"]),
            "processing": any(term in pred_str for term in ["process", "clean", "transform", "filter"]),
            "analysis": any(term in pred_str for term in ["analyz", "pattern", "insight", "trend"]),
            "visualization": any(term in pred_str for term in ["visual", "chart", "plot", "map", "graph"])
        }
        
        # Calculate score based on components present
        score = sum(components.values()) / len(components)
        
        # Bonus for specific insights mentioned
        if isinstance(expected, dict) and "insights" in expected:
            expected_insights = expected["insights"].lower()
            insight_keywords = ["correlation", "increase", "decrease", "pattern", "trend", "risk"]
            insight_score = sum(1 for keyword in insight_keywords if keyword in pred_str) / len(insight_keywords)
            score = (score + insight_score) / 2
        
        return score
    
    def optimize(self):
        """Run the optimization process"""
        print("🔧 Loading training data...")
        examples = self.load_training_data()
        
        print(f"📊 Loaded {len(examples)} training examples")
        
        # Split data
        train_size = int(len(examples) * 0.8)
        trainset = examples[:train_size]
        testset = examples[train_size:]
        
        print(f"🎯 Training on {len(trainset)} examples, testing on {len(testset)}")
        
        # Set up optimizer
        optimizer = dspy.MIPROv2(
            metric=self.workflow_quality_metric,
            num_candidates=15,
            init_temperature=1.0
        )
        
        # Optimize the pipeline
        print("⚡ Starting optimization...")
        optimized_pipeline = optimizer.compile(
            self.pipeline,
            trainset=trainset,
            num_trials=30,
            max_bootstrapped_demos=6
        )
        
        # Test performance
        print("📈 Testing optimized pipeline...")
        test_scores = []
        for example in testset:
            try:
                prediction = optimized_pipeline(user_request=example.user_request)
                score = self.workflow_quality_metric(prediction, example)
                test_scores.append(score)
            except Exception as e:
                print(f"❌ Test failed: {e}")
                test_scores.append(0.0)
        
        avg_score = sum(test_scores) / len(test_scores) if test_scores else 0.0
        print(f"🎯 Average test score: {avg_score:.3f}")
        
        # Save optimized pipeline
        self.output_path.parent.mkdir(exist_ok=True)
        with open(self.output_path, 'wb') as f:
            import pickle
            pickle.dump(optimized_pipeline, f)
        
        print(f"💾 Optimized pipeline saved to {self.output_path}")
        
        return optimized_pipeline, avg_score
    
    def compare_models(self, models: list):
        """Optimize for multiple models and compare"""
        results = {}
        
        for model in models:
            print(f"\n🔄 Optimizing for {model}...")
            self.model_name = model
            dspy.configure(lm=dspy.LM(model))
            
            try:
                _, score = self.optimize()
                results[model] = score
            except Exception as e:
                print(f"❌ Failed to optimize {model}: {e}")
                results[model] = 0.0
        
        # Show comparison
        print("\n📊 Model Comparison:")
        for model, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model}: {score:.3f}")
        
        return results

# CLI for optimization
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize DSPy Geospatial Pipeline")
    parser.add_argument("--model", default="openai/gpt-4", help="Model to optimize for")
    parser.add_argument("--compare", nargs="+", help="Compare multiple models")
    
    args = parser.parse_args()
    
    optimizer = PipelineOptimizer(args.model)
    
    if args.compare:
        optimizer.compare_models(args.compare)
    else:
        optimizer.optimize()
```

### 6. Deployment and Automation

```bash
#!/bin/bash
# scripts/setup_dspy_only.sh

echo "🚀 Setting up DSPy-only geospatial workflow..."

# Create directory structure
mkdir -p {core,cli,optimization,datasets,models,workspace}

# Install dependencies
pip install dspy-ai langchain-openai matplotlib pandas geopandas requests

# Create sample training data if it doesn't exist
if [ ! -f "datasets/complete_workflows.json" ]; then
    echo "📝 Creating sample training data..."
    cat > datasets/complete_workflows.json << 'EOF'
[
  {
    "user_request": "Analyze earthquake patterns in California",
    "expected_output": {
      "data_collected": "USGS earthquake data for California",
      "analysis": "Pattern analysis showing fault line activity",
      "insights": "Increased activity along San Andreas fault",
      "visualization": "Map and timeline visualizations created"
    }
  }
]
EOF
fi

echo "✅ Setup complete! Run: python cli/simple_cli.py"
```

```bash
#!/bin/bash
# scripts/optimize_and_deploy.sh

MODEL=${1:-"openai/gpt-4"}

echo "🔧 Optimizing pipeline for model: $MODEL"

# Run optimization
python optimization/optimizer.py --model "$MODEL"

if [ $? -eq 0 ]; then
    echo "✅ Optimization complete!"
    echo "🚀 Starting optimized CLI..."
    python cli/simple_cli.py --model "$MODEL"
else
    echo "❌ Optimization failed"
    exit 1
fi
```

## Benefits of DSPy-Only Approach

### 1. **Unified Optimization**
- DSPy optimizes the entire workflow end-to-end
- No fragmentation across framework boundaries
- Learned routing and task decomposition

### 2. **Simpler Architecture**
- Single framework to learn and maintain
- No integration complexity
- Clear responsibility boundaries

### 3. **Natural Composition**
- DSPy modules compose naturally
- Built-in state management
- Consistent error handling

### 4. **Model Agnostic**
- Easy model switching through DSPy
- Automatic re-optimization for new models
- Consistent interface across models

### 5. **End-to-End Learning**
- System learns optimal workflows
- Discovers better task decomposition
- Improves routing decisions over time

## Trade-offs vs DeepAgents CLI

### What We Lose
1. **Built-in File Management**: Need to implement our own
2. **TODO Management**: No automatic task tracking
3. **CLI Infrastructure**: Need to build CLI features
4. **Human-in-the-Loop**: No built-in approval workflows
5. **Rich Middleware**: No pre-built middleware ecosystem

### What We Gain
1. **End-to-End Optimization**: Complete workflow optimization
2. **Learned Patterns**: Discovery of optimal approaches
3. **Simpler Debugging**: Single framework boundaries
4. **Natural Code Generation**: DSPy excels at generating code
5. **Unified Model Management**: Consistent model handling

## When to Use DSPy-Only

### Best For:
- **Code-heavy workflows** (data processing, analysis, visualization)
- **Research and experimentation** where discovering optimal approaches matters
- **Model flexibility** requirements
- **Simple deployment** needs
- **Teams comfortable with code generation** approaches

### Avoid When:
- **Complex user interactions** needed (approvals, complex state)
- **Rich tooling ecosystem** required
- **Traditional agent patterns** preferred
- **Non-technical users** need direct interaction

## Development Workflow

### 1. **Create Training Data**
```bash
# Edit datasets/complete_workflows.json with example workflows
# Include user requests and expected outputs
```

### 2. **Develop Pipeline**
```bash
# Implement core DSPy modules
# Test individual components
python -c "from core.geospatial_pipeline import GeospatialPipeline; p = GeospatialPipeline(); print(p('test request'))"
```

### 3. **Optimize**
```bash
# Run optimization with training data
python optimization/optimizer.py --model openai/gpt-4
```

### 4. **Deploy**
```bash
# Run optimized CLI
python cli/simple_cli.py --model gpt-4
```

### 5. **Model Switching**
```bash
# Optimize for new model
python optimization/optimizer.py --model anthropic/claude-3

# Run with new model
python cli/simple_cli.py --model claude-3
```

## Conclusion

The DSPy-only approach eliminates the complexity of multi-framework integration while leveraging DSPy's core strengths: end-to-end optimization, natural composition, and learned workflows. For geospatial workflows that involve significant code generation and data processing, this approach provides a cleaner, more optimizable solution than trying to integrate LangGraph + DSPy.

The trade-off is losing some of the rich tooling and user interaction features of DeepAgents CLI, but gaining the ability to optimize complete workflows and discover optimal approaches through DSPy's learning capabilities.