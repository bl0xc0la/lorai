# Lora AI Workspace

Lora AI is a sleek, local-first artificial intelligence assistant platform. It features a high-fidelity dark workspace UI, multiple specialized system execution modes, dynamic backend model routing using custom Ollama instances, and time-based security protection (2FA) verified via Google Authenticator.

---

## System Architecture Features

* **Dynamic Model Tiering:** Hot-swap processing routes seamlessly between **Lora Speed** (lightweight execution built on `llama3.2:3b`) and **Lora Pro** (deep engineering reasoning built on `gemma4:latest`).
* **System Core Modes:** On-the-fly prompt adaptation switching profiles between a Standard Assistant, Advanced Code Engineer, or a Creative Context Engine.
* **TOTP Security Gateway:** Full verification blocking layer requiring synchronized cryptographic tokens through standard authenticator apps.
* **Modern Workspace UI:** Fluid, responsive dark layout designed with scannable custom layout configurations.

---

## Installation & Setup

### 1. Clone & Project Structuring
Ensure your repository directory contains the following file locations:
```text
LoraAI/
├── api/
│   └── server.py
└── index.html
