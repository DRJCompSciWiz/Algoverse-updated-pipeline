import openai
import requests
import anthropic
import ollama
import google.generativeai as genai
from dotenv import load_dotenv
import os 

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
        load_dotenv()
        self.api_key = os.getenv("PERPLEXITY_API_KEY")
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

class OpenAIAgent:
    def __init__(self, api_key, system_prompt= """You are an expert AI agent designed to solve physics problems by interacting directly with a physics simulator. You have access to a variety of tools to manipulate objects, query object states (position, velocity, acceleration, etc.), and simulate physics progression through time (step).
    
    Here are some important guidelines for interacting with the environment:
    1) ALWAYS Provide clear reasoning for every action.
    2) ALWAYS return actions formatted as valid JSON arrays of tool calls.
    3) Simulate time progression explicitly using the step function.
    4) Query the object states to give you better context of the environment, it will not automatically tell you this.
                 
    Submit your answer only when confident, using the answer function."""):
        
        """
        Initialize the OpenAIAgent.
        Parameters:
        - api_key: str, your OpenAI API key.
        - system_prompt: str, initial instruction for the AI's role.
        """
        load_dotenv()
        self.api_key = os.getenv("OPENAIAGENT_API_KEY")
        self.api_key = api_key
        self.context = [{"role": "system", "content": system_prompt}]



    def interact(self, user_input):
        """
        Send a message to OpenAI and receive a response.
        
        Parameters:
        - user_input: str, the user's input message.
        
        Returns:
        - AI's response as a string.
        """
        self.context.append({"role": "user", "content": user_input})
        
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=self.context,
            api_key=self.api_key
        )
        
        ai_message = response["choices"][0]["message"]["content"]
        self.context.append({"role": "assistant", "content": ai_message})
        return ai_message
        
    def get_context(self):
        """Return the current conversation context."""
        return self.context
        
    def clear_context(self):
        """Reset the conversation history, keeping the system prompt."""
        self.context = [self.context[0]]

# Example usage
# agent = OpenAIAgent(api_key="your_api_key")
# print(agent.interact("What should I do in this MuJoCo environment?"))

class LlamaAgent:
    def __init__(self, api_key, model_name="llama3", system_prompt="""You are an expert AI agent designed to solve physics problems by interacting directly with a physics simulator. You have access to a variety of tools to manipulate objects, query object states (position, velocity, acceleration, etc.), and simulate physics progression through time (step).

Here are some important guidelines for interacting with the environment:
1) ALWAYS Provide clear reasoning for every action.
2) ALWAYS return actions formatted as valid JSON arrays of tool calls.
3) Simulate time progression explicitly using the step function.
4) Query the object states to give you better context of the environment, it will not automatically tell you this.
             
Submit your answer only when confident, using the answer function."""):
        """
        Initialize the LlamaAgent.
        Parameters:
        - api_key: str, your Llama API key.
        - model_name: str, name of the Llama model (e.g., "llama3", "llama2-7b-chat", etc.)
        - system_prompt: str, initial instruction for the AI's role.
        """
        self.api_key = os.getenv("LLAMA_API_KEY")
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

class GrokAgent:
    def __init__(self, api_key, system_prompt="""You are an expert AI agent designed to solve physics problems by interacting directly with a physics simulator. You have access to a variety of tools to manipulate objects, query object states (position, velocity, acceleration, etc.), and simulate physics progression through time (step).

Here are some important guidelines for interacting with the environment:
1) ALWAYS Provide clear reasoning for every action.
2) ALWAYS return actions formatted as valid JSON arrays of tool calls.
3) Simulate time progression explicitly using the step function.
4) Query the object states to give you better context of the environment, it will not automatically tell you this.
             
Submit your answer only when confident, using the answer function."""):
        """
        Initialize the GrokAgent.
        Parameters:
        - api_key: str, your X (Twitter) API token that has Grok access.
        - system_prompt: str, initial instruction for the AI's role.
        """
        self.api_key = os.getenv("GROK_API_KEY")
        self.system_prompt = system_prompt
        self.context = []
        self.api_url = "https://api.grok.x.ai/v1/chat/completions"  # Hypothetical endpoint
    
    def interact(self, user_input):
        """
        Send a message to Grok and receive a response.
        
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
            "model": "grok-1",  # or "grok-1.5" depending on version
            "messages": messages
        }
        
        response = requests.post(self.api_url, headers=headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"Error from Grok API: {response.text}")
        
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
# agent = GrokAgent(api_key="your_x_api_key")
# print(agent.interact("What should I do in this MuJoCo environment?"))

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
        self.api_key = os.getenv("GEMINI_API_KEY")
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
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
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