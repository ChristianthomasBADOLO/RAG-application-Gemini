
import google.generativeai as genai

from config.globals import GOOGLE_AI_API_KEY


# Configure GenAI
genai.configure(api_key=GOOGLE_AI_API_KEY)


class GoogleGenerativeModelInit:
  def __init__(self, model_name:str):
    """
      model_name: str - The name of the model to initialise. enum: ["gemini-pro", "gemini-vision"]
    """
    self.model_name = model_name
    self.api_key = GOOGLE_AI_API_KEY
    
    # Initialise the model
    self.model = genai.GenerativeModel(self.model_name)
    

class GeminiProModelChat(GoogleGenerativeModelInit):
  def __init__(self):
    super().__init__("gemini-pro")
    
    # Start a chat
    self.chat = self.model.start_chat(history=[])
    
  def get_gemini_response(self, prompt:str, stream:bool=False):
    # Generate a response
    response = self.chat.send_message(prompt, stream=stream)
    if not stream:
      return response.text
    return response

