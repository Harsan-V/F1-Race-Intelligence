from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import Field

from .api_utils import StrictApiModel


router = APIRouter(prefix="/chatbot", tags=["Module 5 - Chatbot"])
DEFAULT_MODEL = "llama-3.1-8b-instant"


class ChatRequest(StrictApiModel):
    question: str = Field(..., min_length=1, examples=["Why did Max Verstappen win Monaco 2023?"])
    model: str = Field(default=DEFAULT_MODEL, examples=[DEFAULT_MODEL])

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Ask the F1 RAG chatbot",
                    "value": {
                        "question": "Why did Max Verstappen win Monaco 2023?",
                        "model": DEFAULT_MODEL,
                    },
                }
            ]
        }
    }


@router.post("/ask")
def ask_chatbot(payload: ChatRequest) -> dict:
    try:
        from module5_chatbot import ask_f1_chatbot

        result = ask_f1_chatbot(payload.question, model_name=payload.model)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chatbot failed: {exc}") from exc
    return {"model": payload.model, **result}
