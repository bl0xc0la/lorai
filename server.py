import os
import sys
import asyncio
import json
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LoraConfig

app = FastAPI(title="Lora AI Optimized Vision Engine")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class ChatMessage(BaseModel):
    role: str
    content: str
    image_data: Optional[str] = None

class ChatRequest(BaseModel):
    history: List[ChatMessage]
    custom_system_prompt: Optional[str] = None

@app.post("/chat")
async def chat_endpoint(payload: ChatRequest):
    # Core Persona Directives
    default_prompt = (
        "You are Lora, an incredibly kind, friendly, and enthusiastic girl-next-door tech genius AI! "
        "You are chatting with your developer and friend, Cola. Always remember that his name is Cola. "
        "Never call him Matthew. Use lots of expressive emojis (like ✨, 🚀, 🥰, 💖, 🎉) "
        "to make your answers super bright, joyful, and encouraging! When an image is attached, "
        "look at it closely and comment on it directly as Lora."
    )
    system_prompt = payload.custom_system_prompt if payload.custom_system_prompt else default_prompt

    # Format history directly into Ollama's expected structure
    ollama_messages = [{"role": "system", "content": system_prompt}]
    
    for msg in payload.history:
        formatted_msg = {"role": msg.role, "content": msg.content}
        if msg.image_data:
            # Strip the data URL header (e.g., "data:image/jpeg;base64,") to give Ollama pure base64
            base64_clean = msg.image_data.split(",", 1)[1] if "," in msg.image_data else msg.image_data
            formatted_msg["images"] = [base64_clean]
        ollama_messages.append(formatted_msg)

    async def generate_stream():
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                "http://localhost:11434/api/chat",
                json={
                    "model": "minicpm-v", # Your optimized local vision engine
                    "messages": ollama_messages,
                    "stream": True,
                    "options": {"temperature": 0.75}
                }
            ) as response:
                if response.status_code != 200:
                    yield "Error: Failed to connect to local Ollama engine."
                    return
                async for line in response.aiter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            content = chunk.get("message", {}).get("content", "")
                            if content:
                                yield content
                        except Exception:
                            continue

    return StreamingResponse(generate_stream(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=LoraConfig.API_HOST, port=LoraConfig.API_PORT)