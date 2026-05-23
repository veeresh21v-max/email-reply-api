# AI Email Reply Generator

A FastAPI application that analyzes customer emails and generates 
professional reply drafts using LLM.

## What it does
- Detects sentiment (POSITIVE, NEGATIVE, NEUTRAL)
- Classifies intent (COMPLAINT, INQUIRY, APPRECIATION)
- Detects urgency (HIGH, MEDIUM, LOW)
- Generates professional 3-sentence reply
- Adjusts tone and temperature based on email category
- Escalation alert for HIGH urgency emails

## Tech Stack
- FastAPI — API framework
- Groq + Llama 3.3 70b — LLM
- Pydantic — request/response validation

## Setup
pip install -r requirements.txt
Add GROQ_API_KEY to .env
uvicorn main:app --reload

## Endpoints
- POST /analyze — analyze email only
- POST /generate-reply — analyze and generate reply
- POST /analyze-and-reply — full pipeline (main endpoint)
- GET /health — health check

## Example Request
POST /analyze-and-reply
{
  "email_text": "I ordered a laptop 2 weeks ago and it still has not arrived."
}

## Example Response
{
  "email_text": "...",
  "analysis": {
    "sentiment": "NEGATIVE",
    "category": "COMPLAINT",
    "urgency": "HIGH",
    "key_issue": "Order not arrived after 2 weeks",
    "suggested_tone": "empathetic and apologetic"
  },
  "reply": {
    "reply_text": "...",
    "temperature": 0.3,
    "input_tokens": 186,
    "output_tokens": 94
  },
  "total_tokens": 280
}
