from __future__ import annotations
import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import xml.etree.ElementTree as ET
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)s] | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class Simulator:
    def __init__(self, scene_id: str):
        """
        Initialize the Simulation class with the provided scene ID.
        The model is automatically loaded based on the scene_id.
        """
        self.scene_id = scene_id
        self.model_path = self.get_model_path(scene_id)
        
        try:
            # Load model with error handling
            self.model = mujoco.MjModel.from_xml_path(self.model_path)
            self.data = mujoco.MjData(self.model)
            
            # Initialize viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.start_pos = np.copy(self.data.qpos)
            self.time = 0
            self.prev_velocities = {}  # Store previous velocities for acceleration calculations
            
            # Ensuring the initial velocity is zero
            self.data.qvel[:] = 0.0  # Setting all initial velocities to zero
            
        except Exception as e:
            logging.error(f"MuJoCo initialization failed: {e}")
            raise
    
    #These are functions that help set up the scene for the LLM to interact with

    def get_model_path(self, scene_id: str) -> str:
        """Generate the model path based on the scene_id (returns a string)"""
        try:
            # Get the directory of the current script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            scenes_dir = os.path.join(script_dir, "Scenes")

            # Extract scene number from scene_id, e.g. "1.1"
            scene_number = scene_id.split("_")[-1]

            # Split scene_number into parts, e.g. ["1", "1"]
            parts = scene_number.split(".")
            if len(parts) == 2:
                first_level = f"Scene{parts[0]}"
                second_level = f"Scene{scene_number}"
                xml_path = os.path.join(scenes_dir, first_level, second_level, f"scene{scene_number}.xml")
            else:
                # Fallback to original path if no nested structure
                xml_path = os.path.join(scenes_dir, f"Scene{scene_number}", f"scene{scene_number}.xml")

            # Debugging log to verify the constructed path
            logging.debug(f"Constructed model path: {xml_path}")

            # Verify if the file exists
            if not os.path.exists(xml_path):
                raise FileNotFoundError(f"Scene XML not found at: {xml_path}")

            return xml_path.replace("\\", "/")
        except Exception as e:
            logging.error(f"Path construction failed: {e}")
            raise


    def load_scene(self, scene_id: str):
        try:
            if hasattr(self, 'viewer') and self.viewer is not None:
                self.viewer.close()

            scene_id = str(scene_id)  # ensure it's a string
            self.model_path = self.get_model_path(scene_id)
            logging.info(f"Loading model from: {self.model_path}")

            self.model = mujoco.MjModel.from_xml_path(self.model_path)
            self.data = mujoco.MjData(self.model)

            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.start_pos = np.copy(self.data.qpos)
            self.time = 0

        except Exception as e:
            logging.error(f"Failed to load scene {scene_id}: {e}")

    def render(self):
        """Render the current simulation frame (returns nothing specific)"""
        self.viewer.sync()
        return self.viewer.capture_frame()
        
    def get_body_id(self, object_id: str) -> int:
        """Map object_id to a body ID in the MuJoCo model."""
        # Ensure object_id is a string, then map to the body index
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, object_id)
        
        if body_id == -1:
            raise ValueError(f"Body with name '{object_id}' not found in the scene.")
        
        return body_id

    
    def list_objects(self) -> dict:
        """
        Returns a list of all body names currently in the simulation.
        Allows the LLM to discover which object IDs are valid for tool calls.
        """
        body_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in range(self.model.nbody)
        ]
        # Filter out None or empty names if they exist
        valid_names = [name for name in body_names if name]
        return {"body_names": valid_names}
    
    def get_geom_id(self, object_id: str) -> int:
        """Return the geometry ID corresponding to the object_id."""
        try:
            # Assuming that object_id corresponds to a body or a specific geometry.
            body_id = self.get_body_id(object_id)
            geom_id = self.model.body_geomadr[body_id]
            return geom_id
        except Exception as e:
            logging.error(f"Error in get_geom_id for object_id='{object_id}': {str(e)}")
            raise

    def get_parameters(self, object_id: str) -> dict:
        """Retrieve parameters of an object, respecting scene-defined permissions."""
        object_id = str(object_id)

        # Map the object_id to a corresponding body ID index
        body_id = self.get_body_id(object_id)  # This function maps object_id to body_id

        # Check permissions for parameter access
        permissions = getattr(self, 'permissions', {}).get(object_id, {})
        if not permissions.get("get_parameters", True):  # Default to allowed
            raise PermissionError(f"Access to parameters of object with ID {object_id} is not allowed.")

        try:
            # Now use body_id to access the correct attributes
            return {
                "mass": float(self.model.body_mass[body_id]),  # Use body_id instead of object_id
                "bounding_box": self.model.body_inertia[body_id].tolist(),
                "type": int(self.model.body_parentid[body_id])
            }
        except KeyError as e:
            # Handle case where the object is not found in the model
            raise ValueError(f"Object with ID {object_id} not found in the model: {e}")


    def step(self, duration: float = 1.0):
        """Step the simulation forward by a specified duration."""
        num_steps = int(duration / self.model.opt.timestep)
        remaining_time = duration - (num_steps * self.model.opt.timestep)
        
        for _ in range(num_steps):
            # Perform the simulation step
            mujoco.mj_step(self.model, self.data)
            
            # Ensure the simulation state is updated
            mujoco.mj_forward(self.model, self.data)

            if self.viewer is not None:
                self.viewer.sync()

        if remaining_time > 0:
            # Final step for remaining time if any
            mujoco.mj_step(self.model, self.data)
            
            # Ensure the simulation state is updated
            mujoco.mj_forward(self.model, self.data)
            
            if self.viewer is not None:
                self.viewer.sync()

        self.time += duration
        logging.info(f"Simulation time: {self.time} seconds")

    def reset_sim(self):
        """Reset the simulation to its initial state."""
        self.data.qpos[:] = self.start_pos
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.time = 0

    def __del__(self):
        """Clean up resources when the Simulator object is destroyed."""
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.viewer.close()

    #These are the actual functions that the LLM will use to interact with the scene

    def get_position(self, object_id: str) -> dict:
        """Retrieve the position of an object in the simulation."""
        try:
            # Ensure object_id is a string
            object_id = str(object_id)
            
            # Get body ID based on the object_id
            body_id = self.get_body_id(object_id)
            
            # Ensure body_id is valid before proceeding
            if body_id < 0 or body_id >= len(self.data.xpos):
                raise IndexError(f"Invalid body_id {body_id}. Unable to fetch position.")
            
            # Get the position of the body (assuming xpos stores 3D position data)
            pos = self.data.xpos[body_id]  # [x, y, z]
            
            # Return the position along with the current simulation time
            return {"position": pos.tolist(), "time": self.data.time}

        except Exception as e:
            # Log error with more context for debugging
            logging.error(f"Error in get_position for object_id='{object_id}': {str(e)}")
            return {"error": str(e)}
        
    def change_position(self, object_id: str, dx: float, dy: float, dz: float, in_world_frame: bool) -> dict:
        try:
            object_id = str(object_id)  # Ensure object_id is a string
            body_id = self.get_body_id(object_id)  # Get body ID based on object_id
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{object_id}_joint")
            if joint_id == -1:
                return {"error": f"No joint named {object_id}_joint"}
            
            joint_qpos_addr = self.model.jnt_qposadr[joint_id]
            if in_world_frame:
                self.data.qpos[joint_qpos_addr:joint_qpos_addr+3] += np.array([dx, dy, dz])
            else:
                self.data.qpos[joint_qpos_addr] += dx
                self.data.qpos[joint_qpos_addr+1] += dy
                self.data.qpos[joint_qpos_addr+2] += dz
            
            mujoco.mj_forward(self.model, self.data)
            return {"new_position": self.data.qpos[joint_qpos_addr:joint_qpos_addr+3].tolist()}
        except Exception as e:
            return {"error": str(e)}
        
    def get_displacement(self, object_id: str) -> dict:
        try:
            object_id = str(object_id)  # Ensure object_id is a string
            body_id = self.get_body_id(object_id)  # Get body ID based on object_id
            
            # Ensure xpos and start_pos are numpy arrays for element-wise operations
            current_pos = np.array(self.data.xpos[body_id])  # Current position
            if not hasattr(self, 'start_pos') or len(self.start_pos) < 3:
                raise ValueError("Start position (self.start_pos) has not been properly initialized.")
            
            start_pos = np.array(self.start_pos[:3])  # Ensure it's an array and use the first 3 components
            
            # Calculate displacement (distance between current and start position)
            displacement = np.linalg.norm(current_pos - start_pos)
            
            return {"displacement": float(displacement)}
        
        except Exception as e:
            logging.error(f"Error in get_displacement for object_id='{object_id}': {str(e)}")
            return {"error": str(e)}

    def move_object(self, object_id: str, x: float, y: float, z: float) -> dict:
        try:
            object_id = str(object_id)  # Ensure object_id is a string
            body_id = self.get_body_id(object_id)  # Get body ID based on the object name

            # Get the joint corresponding to the object and set its position
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{object_id}_joint")
            if joint_id == -1:
                return {"error": f"No joint named {object_id}_joint"}

            joint_qpos_addr = self.model.jnt_qposadr[joint_id]
            new_pos = np.array([x, y, z])

            # Optional: safety check
            if np.any(np.isnan(new_pos)) or np.any(np.isinf(new_pos)):
                return {"error": "Invalid position values (NaN or Inf)"}

            self.data.qpos[joint_qpos_addr:joint_qpos_addr+3] = new_pos
            mujoco.mj_forward(self.model, self.data)

            actual_position = self.data.qpos[joint_qpos_addr:joint_qpos_addr+3]
            logging.debug(f"Moved {object_id} to position: {actual_position}")
            return {"position": tuple(actual_position)}

        except Exception as e:
            return {"error": str(e)}
        
    def set_velocity(self, object_id: str, velocity_vector: list) -> dict:
        try:
            if not object_id.startswith("object_"):
                object_id = f"object_{object_id}"
            body_id = self.get_body_id(object_id)
            if body_id == -1:
                return {"error": f"Body named {object_id} not found."}
            if len(velocity_vector) != 3:
                raise ValueError("velocity_vector must be a list of 3 elements")
            self.data.qvel[body_id * 6: body_id * 6 + 3] = velocity_vector
            mujoco.mj_forward(self.model, self.data)
            return {"status": "velocity_set", "object_id": object_id, "velocity": velocity_vector}
        except Exception as e:
            return {"error": f"Failed to set velocity: {str(e)}"}
                
    def get_velocity(self, object_id: str) -> dict:
        """Fetch the linear velocity of a given object in the simulation."""
        try:
            # Ensure object_id is a string and starts with 'object_'
            if not isinstance(object_id, str) or not object_id.startswith("object_"):
                return {"error": f"Invalid object_id format: {object_id}. Expected format: 'object_#'"}
            
            # Get the body ID based on the object_id
            body_id = self.get_body_id(object_id)

            # Retrieve the linear velocity (v) and angular velocity (w) for the body
            # The `cvel` array contains [linear velocity (v), angular velocity (w)]
            # Linear velocity components are stored in the first 3 elements for each body
            linear_velocity = self.data.cvel[body_id * 6: body_id * 6 + 3]  # Linear velocity in 3D

            return {"velocity": linear_velocity.tolist()}

        except Exception as e:
            logging.error(f"Error in get_velocity for object_id='{object_id}': {str(e)}")
            return {"error": str(e)}

    def compute_force(self, object_id: str, mass: float) -> dict:
        """
        Compute the force on an object using F = ma.
        Relies on current acceleration from simulation data.
        """
        try:
            object_id = str(object_id)
            acceleration = self.get_acceleration(object_id)

            force = {
                "x": mass * acceleration["x"],
                "y": mass * acceleration["y"],
                "z": mass * acceleration["z"]
            }
            return force
        except Exception as e:
            return {"error": f"Failed to compute force: {str(e)}"}
        
    def get_torque(self, object_id: str):
        """Calculate the torque acting on an object."""
        try:
            # Ensure object_id is a string, and handle conversion if it's an integer
            object_id = str(object_id)

            # Convert object_id to an integer if it's valid
            try:
                obj_index = int(object_id)
            except ValueError:
                raise ValueError(f"Invalid object_id format: {object_id}. Expected an integer ID.")
            
            # Ensure the index is within the bounds of the qfrc_applied array
            start_index = obj_index * 6 + 3
            end_index = obj_index * 6 + 6

            if start_index >= len(self.data.qfrc_applied) or end_index > len(self.data.qfrc_applied):
                raise IndexError(f"Index out of bounds: {start_index}-{end_index}.")
            
            # Extract the torque values
            torque = self.data.qfrc_applied[start_index:end_index]
            torque_dict = {"x": torque[0], "y": torque[1], "z": torque[2]}
            
            return {"torque": torque_dict}
        
        except Exception as e:
            logging.error(f"Error in get_torque for object_id='{object_id}': {str(e)}")
            return {"error": str(e)}
        
    def apply_force(self, object_id: str, force_vector: list) -> dict:
        """Apply a force to an object."""
        try:
            object_id = str(object_id)  # Ensure object_id is a string
            body_id = self.get_body_id(object_id)  # Get the body ID based on object_id
            
            # Ensure the force_vector has exactly 3 elements (x, y, z components)
            if len(force_vector) != 3:
                raise ValueError("force_vector must be a list of 3 elements: [fx, fy, fz]")

            # Apply the force to the object (set the force vector in the xfrc_applied array)
            self.data.xfrc_applied[body_id, :3] = force_vector  # Apply the force (first 3 elements)

            return {"status": "force_applied", "object_id": object_id, "force": force_vector}
        
        except ValueError as ve:
            return {"error": str(ve)}
        except Exception as e:
            return {"error": f"Failed to apply force to object '{object_id}': {str(e)}"}
        
    def apply_torque(self, object_id: str, torque_vector: list) -> dict:
        """Apply a torque to an object."""
        try:
            object_id = str(object_id)  # Ensure object_id is a string
            body_id = self.get_body_id(object_id)  # Get the body ID based on object_id
            
            # Ensure the torque_vector has exactly 3 elements (torque around x, y, z axes)
            if len(torque_vector) != 3:
                raise ValueError("torque_vector must be a list of 3 elements: [tx, ty, tz]")

            # Apply the torque to the object (set the torque vector in the xfrc_applied array)
            self.data.xfrc_applied[body_id, 3:6] = torque_vector  # Apply the torque (last 3 elements)

            return {"status": "torque_applied", "object_id": object_id, "torque": torque_vector}
        
        except ValueError as ve:
            return {"error": str(ve)}
        except Exception as e:
            return {"error": f"Failed to apply torque to object '{object_id}': {str(e)}"}
        
    def get_kinetic_energy(self, object_id: str, mass: float) -> dict:
        """Calculate the kinetic energy of an object."""
        try:
            object_id = str(object_id)  # Ensure object_id is a string
            
            # Get velocity of the object
            velocity = self.get_velocity(object_id)
            
            if "velocity" not in velocity:
                raise ValueError(f"Could not retrieve velocity for object {object_id}.")
            
            # Ensure velocity is a numpy array for consistent behavior
            velocity_array = np.array(velocity["velocity"])
            
            # Calculate kinetic energy: 0.5 * mass * v^2
            kinetic_energy = 0.5 * mass * np.sum(velocity_array**2)
            
            return {"kinetic_energy": kinetic_energy}

        except Exception as e:
            # Log the error and provide more context for debugging
            logging.error(f"Error in get_kinetic_energy for object_id='{object_id}': {str(e)}")
            return {"error": str(e)}

    def get_potential_energy(self, object_id: str, mass: float, gravity: float = 9.81) -> dict:
        """Calculate the potential energy of an object."""
        try:
            object_id = str(object_id)  # Ensure object_id is a string
            
            # Get position of the object
            position = self.get_position(object_id)
            
            if "position" not in position:
                raise ValueError(f"Could not retrieve position for object {object_id}.")
            
            pos = position["position"]
            
            # Ensure position has at least 3 components (x, y, z)
            if len(pos) < 3:
                raise ValueError(f"Position for object {object_id} is incomplete. Expected [x, y, z], got {pos}.")
            
            # Calculate potential energy: PE = mass * gravity * height (z-axis)
            potential_energy = mass * gravity * pos[2]  # Using z as height
            
            return {"potential_energy": potential_energy}

        except Exception as e:
            # Log the error and provide more context for debugging
            logging.error(f"Error in get_potential_energy for object_id='{object_id}': {str(e)}")
            return {"error": str(e)}
        
    def get_rotational_energy(self, object_id: str, mass: float) -> dict:
        """Calculate the rotational energy of an object."""
        try:
            angular_velocity = self.get_angular_momentum(object_id, mass)["angular_momentum"]
            inertia = self.model.body_inertia[self.get_body_id(object_id)].tolist()
            rotational_energy = 0.5 * np.dot(angular_velocity, inertia)
            return {"rotational_energy": rotational_energy}
        except Exception as e:
            return {"error": str(e)}

    def get_momentum(self, object_id: str, mass: float):
        """Calculate the linear momentum of an object."""
        try:
            object_id = str(object_id)  # Ensure object_id is a string
            
            # Get the velocity of the object
            velocity = self.get_velocity(object_id)
            
            if "velocity" not in velocity:
                raise ValueError(f"Could not retrieve velocity for object {object_id}.")
            
            vel = velocity["velocity"]
            
            # Ensure velocity has at least 3 components (x, y, z)
            if len(vel) < 3:
                raise ValueError(f"Velocity for object {object_id} is incomplete. Expected [vx, vy, vz], got {vel}.")
            
            # Calculate momentum: p = m * v
            momentum = {
                "x": mass * vel[0],
                "y": mass * vel[1],
                "z": mass * vel[2]
            }
            
            return {"momentum": momentum}

        except Exception as e:
            # Log the error and provide more context for debugging
            logging.error(f"Error in get_momentum for object_id='{object_id}': {str(e)}")
            return {"error": str(e)}

    def get_angular_momentum(self, object_id: str, mass: float) -> dict:
        try:
            object_id = str(object_id)  # Ensure object_id is a string
            body_id = self.get_body_id(object_id)  # Get body ID based on object_id

            # Get the position vector of the body
            position = np.array(self.data.xpos[body_id])  # Assuming xpos holds the position (x, y, z)
            
            # Get the angular velocity components (the last 3 components)
            angvel = np.array(self.data.cvel[body_id][3:6])  # Angular velocity in (wx, wy, wz)
            
            # Calculate the angular momentum as cross product of position and momentum (mass * velocity)
            momentum = mass * np.array(self.data.qvel[body_id][:3])  # Linear momentum (mass * velocity)
            ang_momentum = np.cross(position, momentum)  # Cross product for angular momentum
            
            return {"angular_momentum": ang_momentum.tolist()}
        
        except Exception as e:
            return {"error": str(e)}
        
    def detect_collision(self, obj1_id: str, obj2_id: str) -> dict:
        """Detect collision between two objects and apply simple elastic forces."""
        try:
            obj1_id = str(obj1_id)
            obj2_id = str(obj2_id)
            
            # Convert object IDs to geometry indices (you may need a helper function for this)
            geom1_id = self.get_geom_id(obj1_id)
            geom2_id = self.get_geom_id(obj2_id)
            
            for contact in self.data.contact:
                # Check if the contact involves the two objects
                if (contact.geom1 == geom1_id and contact.geom2 == geom2_id) or \
                (contact.geom1 == geom2_id and contact.geom2 == geom1_id):
                    
                    # Apply simple elastic response based on contact normal and distance
                    normal_force = contact.frame[:3] * contact.dist
                    
                    # Apply force in the opposite direction to obj1 and the same direction to obj2
                    self.apply_force(obj1_id, -normal_force)
                    self.apply_force(obj2_id, normal_force)
                    
                    return {"collision_detected": True, "force_applied": normal_force.tolist()}
            
            # If no collision detected
            return {"collision_detected": False}
        
        except Exception as e:
            logging.error(f"Error in detect_collision: {str(e)}")
            return {"error": str(e)}
        
    def get_center_of_mass(self) -> dict:
        """
        Get the center of mass of the entire simulation.
        """
        try:
            com = self.data.subtree_com[0]  # Get center of mass from the first subtree
            return {"center_of_mass": {"x": com[0], "y": com[1], "z": com[2]}}
        except Exception as e:
            return {"error": f"Failed to retrieve center of mass: {str(e)}"}

    def quat_to_rot_matrix(self, q: list[float]) -> dict:
        try:
            q_np = np.array(q)
            mat = np.zeros((3, 3))
            mujoco.mju_matQuat(mat, q_np)
            return {"rotation_matrix": mat.tolist()}
        except Exception as e:
            return {"error": str(e)}
        
    def python_tool(self, code: str):
        try:
            return eval(code)
        except Exception as e:
            return f"Error executing Python code: {str(e)}"
