import google.generativeai as genai

class GeminiAgent:
    def __init__(self, api_key, system_prompt="""You are an expert AI agent designed to solve physics problems by interacting directly with a physics simulator. You have access to a variety of tools to manipulate objects, query object states (position, velocity, acceleration, etc.), and simulate physics progression through time (step).

Here are some important guidelines for interacting with the environment:
1) ALWAYS Provide clear reasoning for every action.
2) ALWAYS return actions formatted as valid JSON arrays of tool calls.
3) Simulate time progression explicitly using the step function.
4) Query the object states to give you better context of the environment, it will not automatically tell you this.
             
Submit your answer only when confident, using the answer function."""):
        """
        Initialize the GeminiAgent.
        Parameters:
        - api_key: str, your Google AI Studio API key.
        - system_prompt: str, initial instruction for the AI's role.
        """
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
        self.system_prompt = system_prompt
        self.context = []  # Store user-assistant history
    
    def interact(self, user_input):
        """
        Send a message to Gemini and receive a response.
        
        Parameters:
        - user_input: str, the user's input message.
        
        Returns:
        - AI's response as a string.
        """
        chat_history = []
        
        # Add system prompt as first message
        if not self.context:
            chat_history.append({"role": "system", "parts": [self.system_prompt]})
        
        # Add previous conversation
        for msg in self.context:
            chat_history.append({"role": msg["role"], "parts": [msg["content"]]})
        
        # Add the new user message
        chat_history.append({"role": "user", "parts": [user_input]})
        
        # Send to Gemini
        response = self.model.generate_content(
            chat_history,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=4096
            )
        )
        
        ai_message = response.text
        self.context.append({"role": "user", "content": user_input})
        self.context.append({"role": "model", "content": ai_message})
        
        return ai_message
    
    def get_context(self):
        """Return the current conversation context."""
        return self.context
    
    def clear_context(self):
        """Reset the conversation history."""
        self.context = []

# Example usage:
# agent = GeminiAgent(api_key="your_api_key")
# print(agent.interact("What should I do in this MuJoCo environment?"))
