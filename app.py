from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import google.generativeai as genai

# Set up the environment variable for the API key
os.environ["GEMINI_API_KEY"] = "YOUR_API_KEY_HERE"

# Configure the Generative AI client
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Create the model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Initialize the model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Initialize the FastAPI app
app = FastAPI()

# Define a Pydantic model for the input data
class InputText(BaseModel):
    input_text: str

@app.post("/generate")
async def generate_text(data: InputText):
    try:
        # Start the chat session
        chat_session = model.start_chat(history=[])
        
        # Send the input message
        response = chat_session.send_message(data.input_text)
        
        # Return the response text
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the Generative AI Text Generation API"}
