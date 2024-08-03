import os
import re
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import google.generativeai as genai
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

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
)

# Define a Pydantic model for the input
class InputModel(BaseModel):
    input_text: str

app = FastAPI()

def sanitize_input(input_text: str) -> str:
    """
    Sanitize the input text by removing or escaping problematic characters.
    """
    # Remove newlines and unwanted characters
    sanitized_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', input_text)
    return sanitized_text

@app.post("/generate")
async def generate_response(input_model: InputModel):
    try:
        # Sanitize the input text
        sanitized_input_text = sanitize_input(input_model.input_text)
        logging.info(f"Sanitized input: {sanitized_input_text}")

        # Start a chat session
        chat_session = model.start_chat(
            history=[
                # Optionally include any initial conversation history here
            ]
        )

        # Send the sanitized input message to the model
        response = chat_session.send_message(sanitized_input_text)

        logging.info(f"Model response: {response.text}")

        return {"response": response.text}

    except genai.errors.GeminiAPIError as e:
        logging.error(f"API error: {e}")
        raise HTTPException(status_code=502, detail="API Error")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Process-Time"] = str(time.time() - request.state.start_time)
    return response

@app.middleware("http")
async def log_request_data(request: Request, call_next):
    request.state.start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - request.state.start_time
    logging.info(f"Request: {request.method} {request.url} - Process time: {process_time:.4f}s")
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
