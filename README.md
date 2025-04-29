# MuJoCo Physics Simulation with OpenAI Agent

## 🚀 **Project Overview**

This project is a simulation-based experiment platform designed to leverage **MuJoCo** physics simulations, **OpenAI agents**, and various physics tools to solve physics problems. It integrates multiple components that allow interaction with 3D physics simulations, stepwise progression, and decision-making based on AI-driven tool calls.

The system uses **MuJoCo**, a physics engine for robotics and simulation, integrated with a **Python-based OpenAI Agent** to solve tasks involving motion, forces, velocities, accelerations, and more. The agent provides reasoning and decision-making steps, interacting with the simulator using a series of tools.

## ⚙️ **System Architecture**

### **Components**

1. **Simulator**: A Python class that integrates the MuJoCo simulator for loading and controlling physics scenes. The simulator provides methods to interact with 3D objects (e.g., moving objects, applying forces, detecting collisions).
   
2. **Scene**: A class that handles scene metadata and object properties. It loads simulation scenes from JSON files and generates corresponding prompts to guide the agent's interaction with the environment.

3. **OpenAI Agent**: An AI agent powered by GPT-4 that interacts with the physics simulation environment. The agent receives tasks in the form of prompts, processes them, and makes decisions by issuing commands (tool calls) to the simulator.

4. **Experimental**: A class that runs the experimental setup. It initializes the scene, simulator, and agent, handles iterations, and logs the results of the experiment.

5. **Main**: A script that initializes the process for running experiments with predefined scenes, aggregates results, and saves them to a JSON file.

6. **Scenes**: A folder containing all the different scenes in the class each in nested folders. Each folder contains different variations for each scene, each folder contains a XML file and a json file.

### **Technologies Used**

- **MuJoCo**: A physics engine for simulating rigid body dynamics.
- **OpenAI GPT-4**: Used for generating reasoning and decision-making based on simulation data.
- **Python**: Core language for implementing the system.
- **JSON**: For data serialization and scene definition.
- **Logging**: For detailed logging of each experiment step.

---

## 📚 **Documentation**

### **Installation**

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/mujoco-openai-agent.git
    cd mujoco-openai-agent
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have **MuJoCo** installed and configured as per the official documentation. You also need an OpenAI API key to use the agent.

### **Running Experiments**

1. **Setup Scene and Simulator:**
    Define the scene and simulation parameters by adjusting the `scene_id` in the `main.py` file. Each scene corresponds to a specific physics scenario.

2. **Run the Experiment:**
    To run an experiment with the predefined scene, execute:
    ```bash
    python main.py
    ```
    This will start the experiment for the specified scene, and you’ll see the experiment output and aggregated results in the console. The results will also be saved in a file named `aggregated_results.json`.

3. **Experiment Workflow:**
    - The agent will interact with the scene step-by-step.
    - The agent queries the simulator, performs actions, and makes decisions based on the information provided.
    - Tool calls are executed by the simulator, and results are logged.

### **Key Methods**

1. **Simulator Methods**:
    - `move_object(object_id: str, x: float, y: float, z: float)`: Moves an object in the scene.
    - `apply_force(object_id: str, force_vector: list)`: Applies a force to an object.
    - `step(duration: float)`: Steps the simulation forward by a given duration.
    - `get_velocity(object_id: str)`: Retrieves the velocity of an object.
    - More methods are available for other physics interactions like calculating kinetic energy, momentum, detecting collisions, etc.

2. **OpenAI Agent**:
    - `interact(user_input: str)`: Sends a message to the OpenAI agent and receives a response.
    - `clear_context()`: Resets the conversation context for a fresh start.

3. **Experimental Class**:
    - `run_experiment()`: Orchestrates the experiment by running multiple iterations and interacting with the simulation.
    - `execute_tool_calls(tool_calls_json: str)`: Executes the tool calls and aggregates results.
    - `extract_json_response(llm_output: str)`: Extracts valid JSON tool calls from the agent’s output.

---

### **Contact**
------------------
