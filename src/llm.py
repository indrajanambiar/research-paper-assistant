import os
import sys
from groq import Groq

class LLMEngine:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = None
        self.model = "llama-3.1-8b-instant"  # Updated to latest supported model
        
    def load_model(self, api_key: str = None):
        """Initializes the Groq client."""
        if api_key:
            self.api_key = api_key
            
        if not self.api_key:
            raise ValueError("Groq API Key is missing. Please provide it in the sidebar or .env file.")
            
        try:
            self.client = Groq(api_key=self.api_key)
            # Test connection
            self.client.models.list()
            print("✅ Groq client initialized successfully")
        except Exception as e:
            print(f"Error initializing Groq client: {e}")
            self.client = None
            raise e

    def generate_response(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.2, stream: bool = False):
        """
        Generates a response using Groq API.
        """
        if not self.client:
            self.load_model()
            
        try:
            # Groq uses standard chat format, but we are passing a pre-constructed prompt.
            # We'll treat the whole prompt as a user message for simplicity, 
            # or better, we should refactor prompts.py to return messages list.
            # For now, passing the prompt as a single user message works for Llama 3.
            
            messages = [{"role": "user", "content": prompt}]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
                stop=None
            )
            
            if stream:
                return response
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            if stream:
                def error_gen():
                    yield {'choices': [{'delta': {'content': f"Error: {error_msg}"}}]} 
                return error_gen()
            return f"Error generating response: {error_msg}"

    def is_model_loaded(self) -> bool:
        return self.client is not None
