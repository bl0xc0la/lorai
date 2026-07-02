// app.js - Unified Frontend Logic for Lora AI

// Update this to match your local server port and endpoint exactly
const API_URL = "http://localhost:8000/chat";

async function sendMessage() {
    const input = document.getElementById("userInput");
    const query = input.value.trim();
    if (!query) return;

    const chatBox = document.getElementById("chatBox");

    // 1. Append User Message to UI
    const userDiv = document.createElement("div");
    userDiv.className = "message user";
    userDiv.innerText = query;
    chatBox.appendChild(userDiv);
    
    // Clear input and scroll down
    input.value = "";
    chatBox.scrollTop = chatBox.scrollHeight;

    try {
        // 2. Send request to the local backend engine
        const response = await fetch(API_URL, {
            method: "POST",
            headers: { 
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ message: query })
        });

        if (!response.ok) throw new Error("Server returned an error status");
        
        const data = await response.json();
        
        // 3. Append AI Response to UI
        const loraDiv = document.createElement("div");
        loraDiv.className = "message lora";
        loraDiv.innerText = data.reply;
        chatBox.appendChild(loraDiv);

    } catch (err) {
        // 4. Handle connection dropouts gracefully
        console.error("Connection error:", err);
        const errorDiv = document.createElement("div");
        errorDiv.className = "message lora";
        errorDiv.style.color = "#ef4444";
        errorDiv.innerText = "Failed to connect to Lora server core engine.";
        chatBox.appendChild(errorDiv);
    }
    
    // Final scroll adjustment
    chatBox.scrollTop = chatBox.scrollHeight;
}

// Allow pressing 'Enter' inside the input box to send the message
document.addEventListener("DOMContentLoaded", () => {
    const userInput = document.getElementById("userInput");
    if (userInput) {
        userInput.addEventListener("keydown", (event) => {
            if (event.key === "Enter") {
                sendMessage();
            }
        });
    }
});
