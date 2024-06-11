import os
from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai

# Configure Gemini API
os.environ["GEMINI_API_KEY"] = "AIzaSyB6nT_Ib5cnSSZgnQpcBialvlcZG7UcJi4"
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Create FastAPI app instance
app = FastAPI()

# Define request body model
class Message(BaseModel):
    input_text: str

# Initialize GenerativeModel
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Define endpoint for chat
@app.post("/chat/")
async def chat(message: Message):
    # Start chat session
    chat_session = model.start_chat(history=[])

    # Send message to chat session
    response = chat_session.send_message(message.input_text)

    # Return response text
    return {"response": response.text}
