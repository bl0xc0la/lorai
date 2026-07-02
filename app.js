const API_URL = "http://localhost:5000"; // Changed from API_BASE_URL to API_URL
const chatForm = document.getElementById("chatForm");
const userInput = document.getElementById("userInput");
const messagesContainer = document.getElementById("messagesContainer");
const themeToggle = document.getElementById("themeToggle");
const newChatBtn = document.getElementById("newChatBtn");
const historyList = document.getElementById("historyList");
const mediaUploadInput = document.getElementById("mediaUploadInput");
const imagePreviewDock = document.getElementById("imagePreviewDock");
const previewImg = document.getElementById("previewImg");
const systemPromptInput = document.getElementById("systemPromptInput");

let conversationHistory = [];
let attachedImageBase64 = null;

// Handle Image Loading Processing File Matrix Loops
mediaUploadInput.addEventListener("change", function(e) {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function(event) {
        attachedImageBase64 = event.target.result;
        previewImg.src = attachedImageBase64;
        imagePreviewDock.style.display = "flex";
    };
    reader.readAsDataURL(file);
});

function clearSelectedImage() {
    attachedImageBase64 = null;
    mediaUploadInput.value = "";
    imagePreviewDock.style.display = "none";
}

themeToggle.addEventListener("click", () => {
    document.body.classList.toggle("light-mode");
    document.body.classList.toggle("dark-mode");
});

newChatBtn.addEventListener("click", () => {
    conversationHistory = [];
    messagesContainer.innerHTML = `
        <div class="message assistant-msg">
            <div class="avatar">L</div>
            <div class="msg-content">New session initialized</div>
        </div>`;
});

function appendMessagePlaceholder(isUser, imageSrc = null) {
    const msgDiv = document.createElement("div");
    msgDiv.classList.add("message", isUser ? "user-msg" : "assistant-msg");
    
    const avatar = document.createElement("div");
    avatar.classList.add("avatar");
    avatar.textContent = isUser ? "C" : "L";
    
    const content = document.createElement("div");
    content.classList.add("msg-content");
    
    if (imageSrc) {
        const imgElement = document.createElement("img");
        imgElement.src = imageSrc;
        imgElement.style.maxWidth = "200px";
        imgElement.style.borderRadius = "8px";
        imgElement.style.marginBottom = "8px";
        imgElement.style.display = "block";
        content.appendChild(imgElement);
    }
    
    msgDiv.appendChild(avatar);
    msgDiv.appendChild(content);
    messagesContainer.appendChild(msgDiv);
    
    return content;
}

window.copyToClipboard = (button) => {
    const codeBlock = button.parentElement.nextElementSibling.querySelector("code");
    navigator.clipboard.writeText(codeBlock.textContent).then(() => {
        button.textContent = "Copied! ✓";
        setTimeout(() => button.textContent = "Copy", 2000);
    });
};

function parseMarkdown(text) {
    const parts = text.split(/(```[\s\S]*?```)/g);
    return parts.map(part => {
        if (part.startsWith("```") && part.endsWith("```")) {
            const lines = part.split("\n");
            const language = lines[0].replace("```", "").trim() || "code";
            const codeContent = lines.slice(1, -1).join("\n");
            return `
                <div class="code-container-block">
                    <div class="code-header-row">
                        <span class="code-lang-tag">${language}</span>
                        <button class="code-copy-btn" onclick="copyToClipboard(this)">Copy</button>
                    </div>
                    <pre class="code-pre-element"><code>${escapeHTML(codeContent)}</code></pre>
                </div>`;
        }
        return part.replace(/\n/g, "<br>");
    }).join("");
}

function escapeHTML(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = userInput.value.trim();
    if (!query && !attachedImageBase64) return;
    
    userInput.value = "";
    
    // Save state before clearing dock inputs
    const currentImg = attachedImageBase64;
    clearSelectedImage();

    conversationHistory.push({ role: "user", content: query, image_data: currentImg });
    
    const userContentNode = appendMessagePlaceholder(true, currentImg);
    const textSpan = document.createElement("span");
    textSpan.textContent = query;
    userContentNode.appendChild(textSpan);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    const assistantContentNode = appendMessagePlaceholder(false);
    assistantContentNode.textContent = "●";

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ 
                history: conversationHistory,
                custom_system_prompt: systemPromptInput.value.trim() || null
            })
        });

        if (!response.ok) throw new Error();

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let accumulatedText = "";

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            accumulatedText += decoder.decode(value, { stream: true });
            assistantContentNode.innerHTML = parseMarkdown(accumulatedText);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        conversationHistory.push({ role: "assistant", content: accumulatedText });

    } catch (err) {
        assistantContentNode.innerHTML = `<span style="color: #ef4444;">Failed to connect to Lora server core engine.</span>`;
    }
});
