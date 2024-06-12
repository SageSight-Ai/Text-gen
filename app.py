import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai

# Set your GEMINI API Key
os.environ["GEMINI_API_KEY"] = "AIzaSyB6nT_Ib5cnSSZgnQpcBialvlcZG7UcJi4"
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Configure the generative model
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
    # Adjust safety settings if needed
    # safety_settings = {}
)

# Define a Pydantic model for the input
class InputModel(BaseModel):
    input_text: str

app = FastAPI()

@app.post("/generate")
async def generate_response(input_model: InputModel):
    try:
        # Start a chat session
        chat_session = model.start_chat(
            history=[
                # Optionally include any initial conversation history here
            ]
        )

        # Send the input message to the model
        response = chat_session.send_message(input_model.input_text)

        return {"response": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
