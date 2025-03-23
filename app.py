from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import os

# Define the FastAPI app
app = FastAPI(title="MediTalk API")

# Add CORS middleware to allow frontend to communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load the mapping file
with open("mapping.json", "r") as f:
    disease_mapping = json.load(f)
    # Create a reverse mapping (id to disease name)
    reverse_mapping = {v: k for k, v in disease_mapping.items()}

# Define the model path - using the latest checkpoint
MODEL_PATH = "results/checkpoint-2118"

# Load the model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()  # Set the model to evaluation mode
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    # We'll continue without the model for now and handle errors in the endpoint

# Define the request model
class ChatRequest(BaseModel):
    message: str

# Define the response model
class ChatResponse(BaseModel):
    response: str
    possible_conditions: list = []

@app.get("/")
async def root():
    return {"message": "Welcome to MediTalk API"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Process the user's message
        user_message = request.message
        
        # Use the model for classification instead of generation
        inputs = tokenizer(user_message, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get the top 3 predictions
            top_probs, top_indices = torch.topk(probabilities[0], k=3)
            
        # Format the response with the top prediction
        top_disease_id = top_indices[0].item()
        top_confidence = top_probs[0].item() * 100
        top_disease_name = reverse_mapping.get(top_disease_id, "Unknown condition")
        
        response = f"Based on your input, my diagnosis is: {top_disease_name} (Confidence: {top_confidence:.1f}%)"
        
        # Get the top 3 possible conditions
        possible_conditions = []
        for i in range(min(3, len(top_indices))):
            disease_id = top_indices[i].item()
            disease_name = reverse_mapping.get(disease_id, "Unknown condition")
            possible_conditions.append(disease_name)
        
        return ChatResponse(response=response, possible_conditions=possible_conditions)
    
    except Exception as e:
        # Fallback response if model fails
        return ChatResponse(
            response=f"I'm analyzing your symptoms. Could you provide more details about your condition?",
            possible_conditions=[]
        )

@app.get("/conditions")
async def get_conditions():
    """Return all possible medical conditions"""
    return {"conditions": list(disease_mapping.keys())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)