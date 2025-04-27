import ollama

class LlamaAgent:
    def __init__(self, model_name="llama3", system_prompt="""You are an expert AI agent designed to solve physics problems by interacting directly with a physics simulator. You have access to a variety of tools to manipulate objects, query object states (position, velocity, acceleration, etc.), and simulate physics progression through time (step).

Here are some important guidelines for interacting with the environment:
1) ALWAYS Provide clear reasoning for every action.
2) ALWAYS return actions formatted as valid JSON arrays of tool calls.
3) Simulate time progression explicitly using the step function.
4) Query the object states to give you better context of the environment, it will not automatically tell you this.
             
Submit your answer only when confident, using the answer function."""):
        """
        Initialize the LlamaAgent.
        Parameters:
        - model_name: str, name of the Llama model (e.g., "llama3", "llama2-7b-chat", etc.)
        - system_prompt: str, initial instruction for the AI's role.
        """
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.context = []  # Store conversation history
        
    def interact(self, user_input):
        """
        Send a message to the Llama model and receive a response.
        
        Parameters:
        - user_input: str, the user's input message.
        
        Returns:
        - AI's response as a string.
        """
        full_prompt = self._build_prompt(user_input)
        
        response = ollama.chat(
            model=self.model_name,
            messages=full_prompt
        )
        
        ai_message = response['message']['content']
        self.context.append({"role": "user", "content": user_input})
        self.context.append({"role": "assistant", "content": ai_message})
        return ai_message
    
    def _build_prompt(self, user_input):
        """
        Helper function to build the full prompt history for Ollama.
        """
        prompt = []
        
        if not self.context:
            prompt.append({"role": "system", "content": self.system_prompt})
        
        prompt.extend(self.context)
        prompt.append({"role": "user", "content": user_input})
        
        return prompt
    
    def get_context(self):
        """Return the current conversation context."""
        return self.context
    
    def clear_context(self):
        """Reset the conversation history."""
        self.context = []

# Example usage:
# agent = LlamaAgent(model_name="llama3")
# print(agent.interact("What should I do in this MuJoCo environment?"))
