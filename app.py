import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai

# Initialize the FastAPI app
app = FastAPI()

# Configure the Generative AI API key
os.environ["GEMINI_API_KEY"] = "AIzaSyB6nT_Ib5cnSSZgnQpcBialvlcZG7UcJi4"
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Create the generative model
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

# Define the input and output data models
class InputMessage(BaseModel):
    message: str

class OutputMessage(BaseModel):
    response: str

# Define the chat endpoint
@app.post("/chat", response_model=OutputMessage)
def chat(input_message: InputMessage):
    try:
        chat_session = model.start_chat(
            history=[]
        )
        response = chat_session.send_message(input_message.message)
        return OutputMessage(response=response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the application using `uvicorn` if the script is executed directly
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
