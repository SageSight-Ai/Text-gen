from fastapi import FastAPI, HTTPException
import os
import google.generativeai as genai

# Set up the environment variable for the API key
os.environ["GEMINI_API_KEY"] = "AIzaSyB6nT_Ib5cnSSZgnQpcBialvlcZG7UcJi4"

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
    # safety_settings = Adjust safety settings if needed
    # See https://ai.google.dev/gemini-api/docs/safety-settings
)

# Initialize the FastAPI app
app = FastAPI()

@app.post("/generate")
async def generate_text(input_text: str):
    try:
        # Start the chat session
        chat_session = model.start_chat(history=[])
        
        # Send the input message
        response = chat_session.send_message(input_text)
        
        # Return the response text
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add a root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Generative AI Text Generation API"}
