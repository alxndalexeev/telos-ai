# Telos AI - Autonomous Self-Improving Agent

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

> **Disclaimer:** This project is experimental and provided as-is, without warranty. Use at your own risk. Using LLM APIs may incur costs.

Telos is an autonomous, LLM-powered, self-improving AI agent designed to become a superhuman full-stack engineer. It operates independently, learns continuously, and improves its own codebase through explicit memory management and structured self-reflection.

## 🚀 Quickstart

```bash
git clone <your-repo-url>
cd telos-ai
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add your API keys to .env
python main.py
```

## 🧠 How It Works

Telos uses a "Memento Pattern"—inspired by stories of memory loss—where each run starts stateless, reconstructs context from external memory, performs a task, and records its actions for the next cycle. This approach embraces LLM statelessness, enabling autonomy, transparency, and resilience.

## 💡 Features
- **Autonomous operation:** Plans, executes, and improves itself without human intervention
- **Self-improvement:** Analyzes logs, generates and applies code, and evolves its architecture
- **Explicit memory:** Uses external files for persistent memory and context
- **Modular architecture:** Easily swap LLMs, memory backends, and tools
- **Automated testing:** Generates and runs tests to ensure code quality

## 📁 Project Structure

```
telos-ai/
├── config/         # Configuration modules
├── core/           # Main agent logic (heart, planner, executor, memory)
├── monitoring/     # System health and metrics
├── logging/        # Centralized logging
├── tools/          # External tool integrations
├── architecture/   # Architecture management
├── memory/         # Persistent memory storage
├── tests/          # Test suite
├── main.py         # Entry point
├── requirements.txt
└── README.md
```
<!-- For a full file tree and deep technical docs, see /docs or the wiki. -->

## 🔒 Security
- **Never commit API keys or sensitive data.** Use the `.env` file (see `.env.example`).
- **Review memory files before sharing or committing.**

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! See `CONTRIBUTING.md` for guidelines.

## 📜 License
This project is licensed under [CC BY-NC 4.0](LICENSE): free for non-commercial use with attribution.