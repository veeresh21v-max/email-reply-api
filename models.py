from pydantic import BaseModel
from typing import Literal

# ── Input model ────────────────────────────────────────────────────────────
class EmailRequest(BaseModel):
    email_text: str
    # Raw customer email text sent by the user

# ── Analysis result ────────────────────────────────────────────────────────
class EmailAnalysis(BaseModel):
    sentiment:      Literal["POSITIVE", "NEGATIVE", "NEUTRAL"]
    category:       Literal["COMPLAINT", "INQUIRY", "APPRECIATION"]
    urgency:        Literal["HIGH", "MEDIUM", "LOW"]
    key_issue:      str
    suggested_tone: str

# ── Reply result ───────────────────────────────────────────────────────────
class EmailReply(BaseModel):
    reply_text:    str
    temperature:   float
    input_tokens:  int
    output_tokens: int

# ── Combined response ──────────────────────────────────────────────────────
class FullResponse(BaseModel):
    email_text:   str
    analysis:     EmailAnalysis
    reply:        EmailReply
    total_tokens: int