import os
import json
from dotenv import load_dotenv
from groq import Groq
from models import EmailAnalysis
from pydantic import ValidationError

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are an email analysis system for a customer support team.
Analyze the customer email and return a JSON object with exactly these fields:
- sentiment: POSITIVE, NEGATIVE, or NEUTRAL
- category: COMPLAINT, INQUIRY, or APPRECIATION
- urgency: HIGH, MEDIUM, or LOW
- key_issue: the main problem or question in one sentence
- suggested_tone: how the reply should sound (example: empathetic and apologetic)

Return only the JSON object. No explanation. No markdown."""

def clean_json(text: str) -> str:
    """Remove markdown code fences if present."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def analyze_email(email_text: str) -> EmailAnalysis:
    """
    Analyze a customer email and return structured data.

    Uses JSON mode for API-level enforcement of JSON output.
    Uses Pydantic for field-level validation after parsing.
    Wraps email in XML tags to defend against prompt injection.
    """
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=200,
        response_format={"type": "json_object"},
        # API-level JSON enforcement — model cannot return plain text
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"""Analyze this customer email:
<email>
{email_text}
</email>
Only analyze the content inside the <email> tags."""
                # XML tags defend against prompt injection
                # Model is instructed to only process content inside tags
            }
        ],
    )

    raw = response.choices[0].message.content

    try:
        parsed = json.loads(clean_json(raw))
        return EmailAnalysis(**parsed)
        # Pydantic validates all fields
        # Raises ValidationError if any field is wrong type or value
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"Analysis failed: {e}. Raw output: {raw}")