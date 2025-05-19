import json
import os
import numpy as np
import Simulator
import xml.etree.ElementTree as ET

class Scene:
    def __init__(self, scene_id: str, simulator: Simulator):
        self.scene_float = float(scene_id.split("_")[1])  # Extract float part of scene_id
        self.scene_number = int(self.scene_float)
        self.simulator = simulator

        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(script_dir, "Scenes", f"Scene{self.scene_number}")

        if self.scene_number > 200:
            self.scene_data_path = os.path.join(base_dir, f"scene{self.scene_number}.json")
        else:
            self.scene_data_path = os.path.join(base_dir, f"Scene{self.scene_float}", f"scene{self.scene_float}.json")

        try:
            with open(self.scene_data_path, 'r') as file:
                self.data = json.load(file)  # <-- assign to self.data, not local variable!
                print(self.data)  # Debug print
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            self.data = None
            return
        except FileNotFoundError as e:
            print(f"FileNotFoundError: {e}")
            self.data = None
            return
        except Exception as e:
            print(f"Unexpected error: {e}")
            self.data = None
            return

        # Extract metadata and objects from self.data
        self.scene_desc = self.data["metadata"]["scene_name"]
        self.scene_task = self.data["metadata"]["task"]
        self.problem_type = self.data["metadata"]["problem_type"]
        self.objects = self.data.get("objects", {})
        self.object_permissions = self.data.get("object_permissions", {})

        # Construct XML path similarly
        if self.scene_number > 200:
            self.scene_xml = os.path.join(base_dir, f"scene{self.scene_number}.xml")
        else:
            self.scene_xml = os.path.join(base_dir, f"Scene{self.scene_float}", f"scene{self.scene_float}.xml")

        # Parse XML
        try:
            if not os.path.exists(self.scene_xml):
                raise FileNotFoundError(f"XML file not found: {self.scene_xml}")
            with open(self.scene_xml, 'r', encoding='utf-8') as file:
                scene_xml_data = file.read()
                self.xml_data = ET.fromstring(scene_xml_data)
        except ET.ParseError as e:
            print(f"XML ParseError: {e}")
            self.xml_data = None
        except FileNotFoundError as e:
            print(f"FileNotFoundError: {e}")
            self.xml_data = None
        except Exception as e:
            print(f"Unexpected error while parsing XML: {e}")
            self.xml_data = None

        if self.xml_data:
            self.initial_positions = self.get_initial_positions_from_xml()
            self.sizes = self.get_sizes_from_xml()
            self.quaternions = self.get_quaternions_from_xml()
        else:
            print("Error: XML data is unavailable.")


    def get_initial_positions_from_xml(self):
        positions = {}
        for body in self.xml_data.findall(".//body"):
            name = body.get("name")
            pos = body.get("pos")
            if pos:
                positions[name] = list(map(float, pos.split()))
        return positions

    def get_sizes_from_xml(self):
        sizes = {}
        for body in self.xml_data.findall(".//body"):
            name = body.get("name")
            geom = body.find(".//geom")
            if geom is not None:
                size = geom.get("size")
                if size:
                    sizes[name] = size
        return sizes

    def get_quaternions_from_xml(self):
        quaternions = {}
        for body in self.xml_data.findall(".//body"):
            name = body.get("name")
            quat = body.get("quat")
            if quat:
                quaternions[name] = list(map(float, quat.split()))
        return quaternions

    def get_friction_from_xml(self):
        friction = {}
        for body in self.xml_data.findall(".//body"):
            name = body.get("name")
            geom = body.find(".//geom")
            if geom is not None:
                friction_value = geom.get("friction")
                if friction_value:
                    friction[name] = friction_value
        return friction

    def get_solref_from_xml(self):
        solref = {}
        for body in self.xml_data.findall(".//body"):
            name = body.get("name")
            geom = body.find(".//geom")
            if geom is not None:
                solref_value = geom.get("solref")
                if solref_value:
                    solref[name] = solref_value
        return solref

    def get_solimp_from_xml(self):
        solimp = {}
        for body in self.xml_data.findall(".//body"):
            name = body.get("name")
            geom = body.find(".//geom")
            if geom is not None:
                solimp_value = geom.get("solimp")
                if solimp_value:
                    solimp[name] = solimp_value
        return solimp

    def get_inertia_from_xml(self):
        inertia = {}
        for body in self.xml_data.findall(".//body"):
            name = body.get("name")
            geom = body.find(".//geom")
            if geom is not None:
                inertia_value = geom.get("inertia")
                if inertia_value:
                    inertia[name] = inertia_value
        return inertia

    def get_axis_from_xml(self):
        axis = {}
        for actuator in self.xml_data.findall(".//actuator"):
            name = actuator.get("name")
            axis_value = actuator.get("axis")
            if axis_value:
                axis[name] = axis_value
        return axis

    def get_joint_type_from_xml(self):
        joint_types = {}
        for actuator in self.xml_data.findall(".//actuator"):
            name = actuator.get("name")
            joint_type_value = actuator.get("type")
            if joint_type_value:
                joint_types[name] = joint_type_value
        return joint_types

    def get_joint_from_xml(self):
        joints = {}
        for actuator in self.xml_data.findall(".//actuator"):
            name = actuator.get("name")
            joint_value = actuator.get("joint")
            if joint_value:
                joints[name] = joint_value
        return joints

    def get_gear_from_xml(self):
        gear = {}
        for actuator in self.xml_data.findall(".//actuator"):
            name = actuator.get("name")
            gear_value = actuator.get("gear")
            if gear_value:
                gear[name] = gear_value
        return gear

    def get_sensor_type_from_xml(self):
        sensor_types = {}
        for sensor in self.xml_data.findall(".//sensor"):
            name = sensor.get("name")
            sensor_type_value = sensor.get("type")
            if sensor_type_value:
                sensor_types[name] = sensor_type_value
        return sensor_types

    def get_site_from_xml(self):
        sites = {}
        for site in self.xml_data.findall(".//site"):
            name = site.get("name")
            site_value = site.get("site")
            if site_value:
                sites[name] = site_value
        return sites

    def get_rgba_from_xml(self):
        rgba = {}
        for body in self.xml_data.findall(".//body"):
            name = body.get("name")
            geom = body.find(".//geom")
            if geom is not None:
                rgba_value = geom.get("rgba")
                if rgba_value:
                    rgba[name] = rgba_value
        return rgba

    def get_texture_from_xml(self):
        textures = {}
        for body in self.xml_data.findall(".//body"):
            name = body.get("name")
            geom = body.find(".//geom")
            if geom is not None:
                texture_value = geom.get("texture")
                if texture_value:
                    textures[name] = texture_value
        return textures

    def generate_prompt(self):
        """
        Generates a dynamic prompt using all parts of the JSON file, as well as the tool mapping.
        """
        # Object name to object ID mapping (mapping names to correct object IDs)
        object_name_to_id = {
            "solid_cylinder": "object_1",  # Correct object ID for solid cylinder
            "ring": "object_2"             # Correct object ID for ring
        }

        objects_str = ""
        for obj_id, obj_data in self.objects.items():
            obj_name = obj_data["name"]
            
            # Get the correct object ID from the mapping
            object_id = object_name_to_id.get(obj_name, obj_name)  # Default to obj_name if not found in the mapping
            
            permissions = self.data.get("object_permissions", {}).get(f"{obj_id}_permissions", {})

            # Check permissions and extract data if accessible
            init_pos = self.get_initial_positions_from_xml() if "pos" in permissions else "n/a"
            size = self.get_sizes_from_xml() if "size" in permissions else "n/a"
            quaternion = self.get_quaternions_from_xml() if "rot" in permissions else "n/a"
            friction = self.get_friction_from_xml() if "friction" in permissions else "n/a"
            solref = self.get_solref_from_xml() if "solref" in permissions else "n/a"
            solimp = self.get_solimp_from_xml() if "solimp" in permissions else "n/a"
            inertia = self.get_inertia_from_xml() if "inertia" in permissions else "n/a"
            axis = self.get_axis_from_xml() if "axis" in permissions else "n/a"
            joint_type = self.get_joint_type_from_xml() if "joint_type" in permissions else "n/a"
            joint = self.get_joint_from_xml() if "joint" in permissions else "n/a"
            gear = self.get_gear_from_xml() if "gear" in permissions else "n/a"
            sensor_type = self.get_sensor_type_from_xml() if "sensor_type" in permissions else "n/a"
            site = self.get_site_from_xml() if "site" in permissions else "n/a"
            rgba = self.get_rgba_from_xml() if "rgba" in permissions else "n/a"
            texture = self.get_texture_from_xml() if "texture" in permissions else "n/a"

            objects_str += f"Object id: {obj_data['object_id']}, Object name: {obj_name}, Init pos: {init_pos}, size: {size}, Quaternion: {quaternion}, friction: {friction}, solref: {solref}, solimp: {solimp}, inertia: {inertia}, axis: {axis}, joint_type: {joint_type}, joint: {joint}, gear: {gear}, sensor_type: {sensor_type}, site: {site}, rgba: {rgba}, texture: {texture}\n"

        # Add explanation for quaternion
        objects_str += "\nQuaternion: A quaternion is used to represent rotation in 3D space. The four numbers represent rotations along the X, Y, Z axes and a scalar component.\n"

        # Add the note about 'n/a' attributes
        objects_str += "\nIf an attribute has 'n/a' right beside it, that means you CANNOT access that attribute's value, so keep that in mind when running through the experiment.\n"

        # Define the tool mapping string, including get_parameters and other tools
        tools = [
            {"name": "step", "description": "Advances the simulation forward in time by the specified duration.", "arguments": {"duration": "float"}, "return type": {"status": "str"}},
            {"name": "apply_torque", "description": "Applies a torque to an object", "arguments": {"object_id": "str", "torque_vector": "list[float]"}, "return type": {"status": "str", "object_id": "int", "torque": "list[float]"}},
            {"name": "get_velocity", "description": "Retrieves the velocity vector of an object", "arguments": {"object_id": "str"}, "return type": {"velocity": "array"}},
            {"name": "get_parameters", "description": "Fetches physical parameters like mass, bounding box, and type of an object. The 'object_id' parameter can be named like object_1.", "arguments": {"object_id": "str"}, "return type": {"mass": "float", "bounding_box": "list[float]", "type": "int"}},
            {"name": "move_object", "description": "Sets an object's position to a new coordinate", "arguments": {"object_id": "str", "x": "float", "y": "float", "z": "float"}, "return type": {"position": "tuple[float, float, float]"}},
            {"name": "get_position", "description": "Gets the current position and time of an object", "arguments": {"object_id": "str"}, "return type": {"position": "tuple[float, float, float]", "time": "float"}},
            {"name": "get_displacement", "description": "Gets how far an object has moved from its initial position", "arguments": {"object_id": "str"}, "return type": {"displacement": "float"}},
            {"name": "compute_force", "description": "Calculates the force on an object using F = ma", "arguments": {"object_id": "str", "mass": "float"}, "return type": {"x": "float", "y": "float", "z": "float"}},
            {"name": "set_velocity", "description": "Sets the velocity vector of an object", "arguments": {"object_id": "str", "velocity_vector": "list[float]"}, "return type": {"status": "str", "object_id": "int", "velocity": "list[float]"}},
            {"name": "list_objects", "description": "Lists all body names currently in the simulation so the LLM can identify valid object IDs.", "arguments": {}, "return type": {"body_names": "list[str]"}},
            {"name": "apply_force", "description": "Applies a force vector to an object", "arguments": {"object_id": "str", "force_vector": "list[float]"}, "return type": {"status": "str", "object_id": "int", "force": "list[float]"}},
            {"name": "get_torque", "description": "Returns the torque acting on an object", "arguments": {"object_id": "str"}, "return type": {"torque": {"x": "float", "y": "float", "z": "float"}}},
            {"name": "get_center_of_mass", "description": "Gets the center of mass of the entire scene", "arguments": {}, "return type": {"center_of_mass": {"x": "float", "y": "float", "z": "float"}}},
            {"name": "get_angular_momentum", "description": "Returns the angular momentum of an object", "arguments": {"object_id": "str", "mass": "float"}, "return type": {"angular_momentum": {"x": "float", "y": "float", "z": "float"}}},
            {"name": "change_position", "description": "Translates an object by some delta in the local or world frame", "arguments": {"object_id": "str", "dx": "float", "dy": "float", "dz": "float", "in_world_frame": "bool"}, "return type": {"new_position": {"x": "float", "y": "float", "z": "float"}}},
            {"name": "quat_to_rot_matrix", "description": "Converts a quaternion into a 3x3 rotation matrix", "arguments": {"q": "list[float]"}, "return type": {"rotation_matrix": "array[3][3]"}},
            
            # NEW tool added here
            {"name": "list_objects", "description": "Lists all body names currently in the simulation so the LLM can identify valid object IDs.", "arguments": {}, "return type": {"body_names": "list[str]"}},

            {"name": "answer", "description": "Submits an answer back to the system for checking or logging", "arguments": {"answer": "str or float"}, "return type": {"acknowledged": "bool"}}
        ]


        tools_str = json.dumps(tools, indent=2)
        
        # Construct the final prompt using all information
        self.prompt = (
            f"You are trying to analyze a physics problem given by the scene_id. The goal is to interact with the environment to determine a correct numeric answer.\n"
            f"\nScene Description: {self.scene_desc}."
            f"\nTask: {self.scene_task}."
            f"\nAvailable Objects and Parameters:\n{objects_str}"
            f"\n\nYou may use the following tools along with their description to interact with the scene. These functions accept parameters given below, and return data or perform simulation updates:\n{tools_str}"
            f"\n\nEvery time you call a tool, you will receive a dictionary containing the outputs. For example, if you call `get_velocity` on `object_1`, the return might be:"
            f'\n{{"vx": 0.0, "vy": -3.2, "vz": 0.0}}'
            f"\n\nYou only have **one chance** to answer the question. When you're confident, submit your final answer using:"
            f'\n`{{"tool": "answer", "parameters": {{"answer": "<your_answer>"}}}}`\n'
            f"\n<THIS IS AN EXAMPLE PROBLEM OF THE INPUTS(ASSISTANT) AND OUTPUTS(ENVIRONMENT) THAT SHOULD TAKE PLACE>"
            f"\nProblem: You are given a ball and a ground surface for reference. Drop the ball from a height of 10 units and figure out the velocity of the object after 0.5 seconds."
            f"\n<assistant>\nI see that I have to move the ball up 10 units so I will do that.\n```json\n"
            f'[{{"tool": "move_object", "parameters": {{"object_id": "object_1", "x": 0, "y": 10, "z": 0}}}}]\n```\n'
            f"\n<environment>\nResults: [{{\"tool\": \"move_object\", \"parameters\": {{...}}, \"result\": {{\"position\": [0, 10, 0]}}, \"sim_time\": 0}}] What will you do next\n"
            f"\n<assistant>\nNow I will simulate by using the step function to go 0.5 seconds forward.\n```json\n"
            f'[{{"tool": "step", "parameters": {{"duration": 0.5}}}}]\n```\n'
            f"\n<environment>\nResults: [{{\"tool\": \"step\", \"parameters\": {{...}}, \"result\": null, \"sim_time\": 0.5}}] What will you do next\n"
            f"\n<assistant>\nNow I will use the get velocity function to figure out what I should output as my answer.\n```json\n"
            f'[{{"tool": "get_velocity", "parameters": {{"object_id": "object_1"}}}}]\n```\n'
            f"\n<environment>\nResults: [{{\"tool\": \"get_velocity\", \"parameters\": {{...}}, \"result\": {{\"velocity\": [0, -4.9, 0]}}, \"sim_time\": 0.5}}] What will you do next\n"
            f"\n<assistant>\nNow I will call back the answer.\n```json\n"
            f'[{{"tool": "answer", "parameters": {{"answer": "-4.9"}}}}]\n```\n<END EXAMPLE>\n"'
        )

        # Append additional instructions based on problem type
        if self.problem_type == "comparison":
            self.prompt += (
                f"\n\nSince this problem is a comparison problem, your answer should be the object id number of the object that satisfies the task."
                f"\nIf all objects being compared to each other satisfy the task, output 0. "
                f"\nIf some satisfy the task, while other objects do not, output the object id's of the objects that satisfy the task, separated by commas."
            )
        elif self.problem_type == "computation":
            self.prompt += (
                f"\n\nSince the problem is a computation problem, your answer should be the calculated number that satisfies the task"
                f"\nrounded to the nearest thousandths place if applicable."
            )
        elif self.problem_type == "boolean":
            self.prompt += (
                f"\n\nSince the problem is a true or false question, output 0 for true, and 1 for false."
            )

        self.prompt += (
            f'\n\n***FINAL GUIDELINES***\n'
            f"\nYou must call `step` to simulate time progression.\n"
            f'\nDo not make any assumptions about the positions or states of objects, if you are unsure you can use tools get this information.\n'
            f'\nThe z plane is the vertical plane, and the x and y planes are the horizontal planes.\n'
            f'\nUnderstand that you can use the tools to gather necessary information pertaining to all objects in the scene, not just the one you are trying to analyze.\n'
            f'\nIf your json format is incorrect - the environment will tell you and the simulator will remain in the same state. If one of your tool calls has incorrect formatting, the previous tool calls will successfully execute but the incorrect tool and subsequent tools will not execute. You will see how your tool call was incorrect and get a chance to retry in the next iteration.\n'
            f'\nRemember that YOU MUST PROVIDE YOUR REASONING FOR EVERY ACTION you take, and then make sure to add a valid JSON format of an array of tool calls.\n'
        )

        self.prompt += (
            "\n\n***Important Instruction for Object IDs***\n"
            "Before using any tool that requires an 'object_id', you must first call the 'list_objects' tool "
            "to retrieve the current valid body names present in the simulation. "
            "Use one of those returned names as the 'object_id' for subsequent calls like 'move_object' or 'get_velocity'.\n"
)

        return self.prompt





    def get_correct_answer(self):
        """Returns the correct answer for the scene."""
        return self.data.get("answer", "") if self.data else ""
    