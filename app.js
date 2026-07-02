from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# THIS IS THE SINGLE FILE containing your HTML/CSS/JS
@app.get("/", response_class=HTMLResponse)
def get_ui():
    return """
    <html>
    <head><title>Lora Core</title></head>
    <body style="background:#0b0f19; color:white;">
        <div id="app">
            <input type="text" id="userInput" placeholder="Ask Lora...">
            <button onclick="sendMessage()">Send</button>
            <div id="response"></div>
        </div>
        <script>
            async function sendMessage() {
                const query = document.getElementById("userInput").value;
                // Sending to the root "/" which matches our @app.post("/") below
                const res = await fetch("/", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({query: query})
                });
                const data = await res.json();
                document.getElementById("response").innerText = data.reply;
            }
        </script>
    </body>
    </html>
    """

# THIS HANDLES THE POST REQUEST AT THE ROOT "/"
@app.post("/")
async def chat_endpoint(data: dict):
    user_query = data.get("query")
    # This is where your AI magic happens
    return {"reply": f"Lora received: {user_query}"}
