import os
from dotenv import load_dotenv
from groq import Groq
from models import EmailAnalysis, EmailReply

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "llama-3.3-70b-versatile"

# Temperature map — controls creativity per category
# COMPLAINT    → 0.3: consistent and professional
#                      angry customers need reliable responses
# INQUIRY      → 0.5: balanced — informative but natural
# APPRECIATION → 0.7: warm and varied — some creativity is fine
TEMPERATURE_MAP = {
    "COMPLAINT":    0.3,
    "INQUIRY":      0.5,
    "APPRECIATION": 0.7,
}

def generate_reply(
    email_text: str,
    analysis: EmailAnalysis
) -> EmailReply:
    """
    Generate a professional email reply based on the analysis.

    email_text : original customer email
    analysis   : structured analysis from analyzer.py

    Returns EmailReply with reply text and token usage.
    """
    # Select temperature based on category
    temperature = TEMPERATURE_MAP.get(analysis.category, 0.5)
    # .get() with default 0.5 handles unexpected categories gracefully

    system_prompt = f"""You are a professional customer support specialist.

Tone to use: {analysis.suggested_tone}
Email category: {analysis.category}
Urgency level: {analysis.urgency}
Key issue: {analysis.key_issue}

Rules you must follow:
- Write exactly 3 sentences — no more, no less
- Never promise specific refund amounts
- Always end with an offer to help further
- Match the tone specified above exactly
- Be professional at all times"""

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=200,
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"""Write a professional reply to this customer email:
<email>
{email_text}
</email>
Write only the reply. No subject line. No greeting label. Just the reply text."""
            }
        ],
    )

    reply_text    = response.choices[0].message.content
    input_tokens  = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens

    return EmailReply(
        reply_text=reply_text,
        temperature=temperature,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )