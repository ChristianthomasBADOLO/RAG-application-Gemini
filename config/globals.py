
import os

from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Environment variables
GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY")

# Global variables
class SPEAKER_TYPES:
  USER = "user"
  BOT = "bot"


initial_prompt = {"role": SPEAKER_TYPES.BOT, "content": "How may I assist you today?"}
