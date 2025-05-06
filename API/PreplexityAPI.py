import requests

class PerplexityAgent:
    def __init__(self, api_key, system_prompt="""You are an expert AI agent designed to solve physics problems by interacting directly with a physics simulator. You have access to a variety of tools to manipulate objects, query object states (position, velocity, acceleration, etc.), and simulate physics progression through time (step).

Here are some important guidelines for interacting with the environment:
1) ALWAYS Provide clear reasoning for every action.
2) ALWAYS return actions formatted as valid JSON arrays of tool calls.
3) Simulate time progression explicitly using the step function.
4) Query the object states to give you better context of the environment, it will not automatically tell you this.
             
Submit your answer only when confident, using the answer function."""):
        """
        Initialize the PerplexityAgent.
        Parameters:
        - api_key: str, your Perplexity AI API key.
        - system_prompt: str, initial instruction for the AI's role.
        """
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.context = []
        self.api_url = "https://api.perplexity.ai/chat/completions"
    
    def interact(self, user_input):
        """
        Send a message to Perplexity AI and receive a response.
        
        Parameters:
        - user_input: str, the user's input message.
        
        Returns:
        - AI's response as a string.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        
        if not self.context:
            messages.append({"role": "system", "content": self.system_prompt})
        
        for msg in self.context:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        messages.append({"role": "user", "content": user_input})
        
        payload = {
            "model": "pplx-7b-chat",  # or "pplx-70b-chat" or "pplx-online" depending on your choice
            "messages": messages
        }
        
        response = requests.post(self.api_url, headers=headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"Error from Perplexity API: {response.text}")
        
        ai_message = response.json()["choices"][0]["message"]["content"]
        
        self.context.append({"role": "user", "content": user_input})
        self.context.append({"role": "assistant", "content": ai_message})
        
        return ai_message
    
    def get_context(self):
        """Return the current conversation context."""
        return self.context
    
    def clear_context(self):
        """Reset the conversation history."""
        self.context = []

# Example usage:
# agent = PerplexityAgent(api_key="your_perplexity_api_key")
# print(agent.interact("What should I do in this MuJoCo environment?"))
