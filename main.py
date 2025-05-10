"""
Experiment Runner Script

This script iterates through a predefined list of scene IDs, runs experiments 
using the `Experimental` class, and aggregates the results. The results are 
stored in a dictionary and optionally saved to a JSON file.

Dependencies:
- openai_agent.py
- scene.py
- simulator.py
- experimental.py

Functions:
- main(): Initializes and runs experiments for each scene ID, stores results, 
  and saves them to a JSON file.

Usage:
Run the script directly to execute experiments and generate results.

Example:
$ python script.py

Output:
- Results for each scene are printed to the console.
- Aggregated results are saved in `aggregated_results.json`.

"""

import os
import json
import logging
# Assuming all files are in the same directory and correctly implemented
from AgentClass import OpenAIAgent, PerplexityAgent, GrokAgent, LlamaAgent, GeminiAgent, AnthropicAgent
from Scene import Scene
from Simulator import Simulator
from Experiment import Experiment
import argparse
import shutil
from datetime import datetime
import random
from dotenv import load_dotenv

load_dotenv()

def initialize_agent(agent_type: str):
    """
    Initialize the correct agent based on the provided agent_type string.
    """
    # Example: Make sure to get the API key for the chosen agent
    if agent_type == "OpenAIAgent":
        api_key = os.getenv("OPENAI_API_KEY")
        return OpenAIAgent(api_key)
    elif agent_type == "PerplexityAgent":
        api_key = os.getenv("PERPLEX_API_KEY")
        return PerplexityAgent(api_key)
    elif agent_type == "LlamaAgent":
        api_key = os.getenv("LLAMA_API_KEY")
        return LlamaAgent(api_key)  # Adjust this line as necessary
    elif agent_type == "GrokAgent":
        api_key = os.getenv("GROK_API_KEY")
        return GrokAgent(api_key)
    elif agent_type == "GeminiAgent":
        api_key = os.getenv("GEMINI_API_KEY")
        return GeminiAgent(api_key)
    elif agent_type == "AnthropicAgent":
        api_key = os.getenv("ANTHROPY_API_KEY")
        return AnthropicAgent(api_key)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
def main():
    """
    Executes experiments for predefined scene IDs, collects results, 
    and saves them to a JSON file.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-python-tool", action="store_true", help="Enable the Python evaluation tool")
    args = parser.parse_args()
    enable_python_tool = random.choice([True, False])
    if args.enable_python_tool:
        print("✅ Python tool enabled")
        # Pass this flag into Experiment or wherever tool setup happens
        # e.g., Experiment(enable_python_tool=True)
    else:
        print("❌ Python tool disabled")

    # Predefined list of scene IDs to iterate through
    scene_ids = ["Scene_101"]  # Replace with actual scene IDs

    # Set the agent type (You can modify this to initialize different agents)
    agent_type = "OpenAIAgent"  # Example: you can change this dynamically to switch agents
    
    # Initialize the agent based on the string (this part is your new dynamic agent initialization)
    agent = initialize_agent(agent_type)
    
    # Initialize a dictionary to store aggregated results
    aggregated_results = {}

    # Set the base directory for test results
    base_dir = os.path.join(os.getcwd(), "TestResults")
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    for scene_id in scene_ids:
        scene = Scene(scene_id)  # Initialize the scene
        
        # Generate scene prompt and print to the terminal
        prompt = scene.generate_prompt()
        print(prompt)
        
        # Simulate randomization of the Python tool (enabled or disabled)
        enable_python_tool = random.choice([True, False])
        print(f"Python tool {'enabled' if enable_python_tool else 'disabled'}")
        
        # Initialize Experiment object with Python tool randomization
        experiment = Experiment(scene_id, max_iterations=5, enable_python_tool=enable_python_tool)
        
        # Run the experiment
        results = experiment.run_experiment()

        # Process results
        if results['answer_found']:
            print("\n=== Answer Summary ===")
            print(f"LLM's Answer: {results['llm_answer']}")
            print(f"Correct Answer: {results['correct_answer']}")
            print(f"Answer Correct: {results['correct']}")
        else:
            print("\nNo answer was provided by the LLM.")
        
        # Store the results in the aggregated_results dictionary
        aggregated_results[scene_id] = results

        # Prepare directories for saving the experiment result files
        scene_number = scene.scene_number
        scene_float = scene.scene_float
        result_dir = os.path.join(base_dir, f"Scene{scene_number}", f"Scene{scene_float}")
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        # Save results to new directory
        json_filename = f"aggregated_results_scene_{scene_float}.json"
        log_filename = f"error_logging_scene_{scene_float}.txt"
        
        # Move and rename .json and log files
        with open(os.path.join(result_dir, json_filename), 'w') as json_file:
            json.dump(results, json_file, indent=4)
        
        with open(os.path.join(result_dir, log_filename), 'w') as log_file:
            log_file.write(f"Experiment Log for Scene {scene_id}\n")
            log_file.write(f"Final Answer: {results['llm_answer']}\n")
        
        # Optionally, clear terminal logs and reset the simulator
        print("\nClearing terminal and resetting simulation...\n")
        os.system('cls' if os.name == 'nt' else 'clear')
        experiment.simulator.reset_sim()
    
    # Optionally, save aggregated results to a global file
    with open("aggregated_results.json", "w") as file:
        json.dump(aggregated_results, file, indent=4)

if __name__ == "__main__":
    main()
    # Set up logging configuration