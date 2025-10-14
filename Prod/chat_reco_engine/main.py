from fastapi import FastAPI
from pydantic import BaseModel,field_validator
import uvicorn
from .chat_engine import chatRecommendationSystem
import logging


logger = logging.getLogger(__name__)
app = FastAPI()
chat_engine=chatRecommendationSystem()

class ChatRequest(BaseModel):
    user_id: int
    text: str

    @field_validator("user_id")
    @classmethod
    def user_id_must_be_positive(cls, v):
        if v < 0:
            raise ValueError("user_id cannot be negative")
        return v

    @field_validator("text")
    @classmethod
    def text_must_be_nonempty(cls, v):
        if not isinstance(v, str) or not v.strip():
            raise ValueError("text must be a non-empty string")
        return v.strip()
    
@app.post("/get_reco")
async def chat(req: ChatRequest):
    try:
        response = chat_engine.return_response(req.user_id, req.text)
        return {"user_id": req.user_id, "response": response}
    except Exception as e:
        logger.exception("Error in return_response")
        return {"user_id": req.user_id,"response": "Sorry, something went wrong while processing your request."}

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)