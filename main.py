import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from models import EmailRequest, EmailAnalysis, FullResponse
from analyzer import analyze_email
from reply_generator import generate_reply

load_dotenv()
app = FastAPI(title="AI Email Reply Generator")

# ── Endpoints ──────────────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze(request: EmailRequest):
    """
    Analyze a customer email.
    Returns sentiment, category, urgency, key issue, suggested tone.
    Does NOT generate a reply — analysis only.
    Useful for routing emails without generating replies.
    """
    if not request.email_text.strip():
        raise HTTPException(status_code=400, detail="Email text cannot be empty.")

    try:
        analysis = analyze_email(request.email_text)
        return analysis
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-reply")
async def generate_reply_endpoint(request: EmailRequest):
    """
    Analyze email and generate a professional reply.
    Returns reply text with token usage.
    """
    if not request.email_text.strip():
        raise HTTPException(status_code=400, detail="Email text cannot be empty.")

    try:
        analysis = analyze_email(request.email_text)
        reply    = generate_reply(request.email_text, analysis)
        return {
            "analysis":    analysis,
            "reply":       reply,
        }
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-and-reply", response_model=FullResponse)
async def analyze_and_reply(request: EmailRequest):
    """
    Main endpoint — combines analysis and reply generation.
    This is the endpoint your Voice Email System would call.

    Flow:
    1. Analyze email — get sentiment, category, urgency
    2. Check urgency — log HIGH urgency emails for escalation
    3. Generate reply — using analysis results
    4. Return complete structured response
    """
    if not request.email_text.strip():
        raise HTTPException(status_code=400, detail="Email text cannot be empty.")

    try:
        # Step 1 — Analyze
        analysis = analyze_email(request.email_text)

        # Step 2 — Escalation check
        # As per Design Q4 — urgency check happens after analysis
        # before reply generation
        if analysis.urgency == "HIGH":
            print(f"[ESCALATION ALERT] HIGH urgency {analysis.category} detected.")
            print(f"Key issue: {analysis.key_issue}")
            # In production: send email/Slack alert to manager team here

        # Step 3 — Generate reply
        reply = generate_reply(request.email_text, analysis)

        # Step 4 — Calculate total tokens
        total_tokens = reply.input_tokens + reply.output_tokens

        return FullResponse(
            email_text=request.email_text,
            analysis=analysis,
            reply=reply,
            total_tokens=total_tokens,
        )

    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "model": "llama-3.3-70b-versatile"}

# Run: uvicorn main:app --reload
# Test: http://127.0.0.1:8000/docs