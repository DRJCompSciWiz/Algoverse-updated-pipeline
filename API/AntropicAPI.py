import anthropic

class AnthropicAgent:
    def __init__(self, api_key, system_prompt="""You are an expert AI agent designed to solve physics problems by interacting directly with a physics simulator. You have access to a variety of tools to manipulate objects, query object states (position, velocity, acceleration, etc.), and simulate physics progression through time (step).

Here are some important guidelines for interacting with the environment:
1) ALWAYS Provide clear reasoning for every action.
2) ALWAYS return actions formatted as valid JSON arrays of tool calls.
3) Simulate time progression explicitly using the step function.
4) Query the object states to give you better context of the environment, it will not automatically tell you this.
             
Submit your answer only when confident, using the answer function."""):
        """
        Initialize the AnthropicAgent.
        Parameters:
        - api_key: str, your Anthropic API key.
        - system_prompt: str, initial instruction for the AI's role.
        """
        self.api_key = api_key
        self.client = anthropic.Anthropic(api_key=api_key)
        self.system_prompt = system_prompt
        self.context = []  # Store user-assistant messages
    
    def interact(self, user_input):
        """
        Send a message to Anthropic and receive a response.
        
        Parameters:
        - user_input: str, the user's input message.
        
        Returns:
        - AI's response as a string.
        """
        self.context.append({"role": "user", "content": user_input})
        
        response = self.client.messages.create(
            model="claude-3-opus-20240229",  # You can change model here
            system=self.system_prompt,
            messages=self.context,
            max_tokens=4096,  # adjust if needed
        )
        
        ai_message = response.content[0].text
        self.context.append({"role": "assistant", "content": ai_message})
        return ai_message
    
    def get_context(self):
        """Return the current conversation context."""
        return self.context
    
    def clear_context(self):
        """Reset the conversation history."""
        self.context = []

# Example usage:
# agent = AnthropicAgent(api_key="your_api_key")
# print(agent.interact("What should I do in this MuJoCo environment?"))
