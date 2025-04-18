# Telos AI - Autonomous Self-Improving Agent

Telos is an autonomous, LLM-powered, self-improving AI agent with the goal of becoming a superhuman full-stack engineer. The project explores how an AI system can operate autonomously, learn continuously, and improve its own capabilities through explicit memory management and structured self-reflection.

## 🎯 Purpose & Vision

Telos represents a new paradigm in autonomous AI agents by:

1. **Embracing LLM Limitations** - Rather than fighting against the stateless nature of LLMs, Telos turns this limitation into a design feature through the "Memento Pattern"

2. **Creating True Autonomy** - Telos operates independently, making decisions, executing tasks, and improving itself without human intervention

3. **Building Long-Term Memory** - Through meticulous external record-keeping, Telos maintains continuity and accumulates knowledge despite the transient nature of LLM inferences

4. **Enabling Self-Evolution** - The system is designed to reflect on its own performance, identify weaknesses, and implement improvements to its own codebase

The long-term vision is to create an AI system that can surpass human capabilities in software engineering while maintaining a transparent, understandable, and controllable development process.

## 💭 The Memento Pattern

Telos is built on a unique cognitive architecture inspired by films like "Memento," "Before I Go to Sleep," and "Paycheck" - stories where the protagonists have memory limitations but compensate through external systems. This directly addresses the fundamental technical limitation of LLMs: their statelessness.

### The Artificial Amnesia Approach

Each time Telos "wakes up" (when its heart beats), it:

1. **Starts with amnesia** - Has no memory of previous states or identity (new LLM inference)
2. **Reconstructs itself** - Reads notes, logs, and identity files to understand who it is (loads context)
3. **Discovers its tasks** - Figures out what it was working on based on breadcrumbs left by its "previous self" (reads task state)
4. **Takes action** - Executes a small, meaningful portion of work (current inference)
5. **Leaves detailed notes** - Documents everything for its "future self" who will wake up with no memory (saves state)
6. **Goes to sleep** - Forgetting everything that just happened (LLM session ends)

This approach embraces the stateless nature of LLMs rather than fighting against it. Like Leonard in "Memento" using notes and tattoos, Telos uses its external memory system to maintain continuity despite having no internal memory between sessions.

### Benefits of the Pattern

- **Philosophical alignment with LLM limitations** - Works with, not against, the stateless nature of LLMs
- **Explicit documentation** - Forces detailed note-taking that improves system transparency
- **Fresh perspective** - Each "awakening" comes with fresh analysis unbiased by previous thinking
- **Natural incremental progress** - Tasks are naturally broken into small, discrete, manageable chunks
- **Resilience** - System can recover gracefully from interruptions or crashes

### Technical & Scientific Value

The Memento Pattern isn't just a philosophical curiosity—it offers practical benefits:

- **Reduced hallucination** - By explicitly loading only relevant context from external storage, we reduce the risk of LLM confabulation
- **Improved reasoning** - Breaking complex tasks into small chunks with explicit state transitions improves reasoning capabilities
- **Enhanced observability** - External memory provides complete transparency into the agent's "thought process"
- **Natural checkpoint system** - Each heartbeat creates a natural recovery point
- **Task persistence** - Long-running tasks can be maintained across system restarts

## 💡 Architecture 

Telos consists of several key components:

1. **Core** - Core system components (`core/`)
   - Heart - The main autonomous loop implementing the Memento pattern (`core/heart.py`)
   - Planner - LLM-powered planning system (`core/planner.py`)
   - Executor - Executes plans and actions (`core/executor.py`)
   - Memory Manager - Telos's external memory system (`core/memory_manager.py`)
   - API Manager - Handles API rate limiting (`core/api_manager.py`)

2. **Monitoring** - System monitoring components (`monitoring/`)
   - Defibrillator - Watchdog process for heart (`monitoring/defibrillator.py`)
   - Resource Monitor - System resource monitoring (`monitoring/resource_monitor.py`)
   - Performance - Performance metrics collection (`monitoring/performance.py`)

3. **Logging** - Logging system for actions and thoughts (`logging/`)
   - Logger - Central logging functionality (`logging/logger.py`)

4. **Tools** - External capabilities like web search (`tools/`)
   - Tavily Search - Integration with search API (`tools/tavily_search.py`)

5. **Architecture** - Architecture management (`architecture/`)
   - Manager - Analyzes and improves system architecture (`architecture/manager.py`)

## 🔧 Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd telos-ai
   ```

2. **Set up Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

   ⚠️ **SECURITY WARNING**: Never commit your `.env` file or API keys to the repository. The `.gitignore` file is configured to exclude these, but always double-check before committing changes.

## 🚀 Running Telos

1. **Start the heart**
   ```bash
   python main.py
   ```

2. **Add tasks**
   - Manually edit `memory/tasks.json` with tasks in the format:
   ```json
   [
     {
       "task": "self-improvement",
       "details": "Analyze logs and suggest improvements"
     },
     {
       "task": "research",
       "details": "Search for information on latest AI trends"
     }
   ]
   ```

## 🤖 Self-Improvement Capability

Telos can improve itself through:

1. **Code Generation** - Can write code for itself using LLM
2. **Code Application** - Can apply generated code to its own source files
3. **Log Analysis** - Can analyze its own logs to identify improvement areas
4. **Task Generation** - Can create new tasks for itself based on analysis
5. **Architecture Management** - Can analyze, propose improvements, and modify its own architecture
6. **Automated Testing** - Can generate, run and analyze tests to ensure code quality and functionality

### Architecture Self-Improvement

The architecture manager enables Telos to:

- Analyze code structure, dependencies, and metrics
- Identify architectural issues like high coupling or low cohesion
- Propose architectural improvements
- Implement changes with safety measures (backups and testing)
- Rollback changes if they cause problems

This capability allows Telos to evolve beyond task-specific scripts and improve its fundamental structure.

### Automated Testing Framework

The testing framework enables Telos to:

- Generate unit tests for individual modules
- Create integration tests for multiple modules
- Run tests across the codebase or for specific modules
- Analyze test coverage to identify untested code
- Generate comprehensive test reports
- Validate changes through systematic testing

This capability ensures that Telos maintains code quality while implementing improvements.

## 📁 Project Structure

```
telos-ai/
├── core/                    # Core system components
│   ├── __init__.py
│   ├── heart.py             # Main autonomous loop
│   ├── planner.py           # Decision-making system
│   ├── executor.py          # Plan execution system
│   ├── memory_manager.py    # Memory management
│   └── api_manager.py       # API rate limiting and management
├── monitoring/              # System monitoring components
│   ├── __init__.py
│   ├── defibrillator.py     # Watchdog process
│   ├── resource_monitor.py  # System resource monitoring
│   └── performance.py       # Performance metrics collection
├── logging/                 # Logging systems
│   ├── __init__.py
│   └── logger.py            # Central logging functionality
├── tools/                   # External tool integrations
│   ├── __init__.py
│   ├── tavily_search.py     # Tavily search integration
│   └── ... other tools
├── architecture/            # Architecture management
│   ├── __init__.py
│   └── manager.py           # Architecture improvement system
├── memory/                  # Persistent memory storage
│   ├── action_log.md        # Log of actions taken
│   ├── thoughts.md          # Log of agent's thoughts
│   ├── tasks.json           # Task queue
│   ├── generated_code/      # LLM-generated code
│   ├── architecture/        # Architecture analysis and backups
│   └── test_results/        # Test execution results
├── tests/                   # Test suite
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   └── framework/           # Test framework utilities
├── config.py                # Global configuration
├── requirements.txt         # Dependencies
├── main.py                  # Entry point script
├── .env.example             # Environment variables example
└── README.md                # Documentation
```

## 🔒 Security Considerations

Since this is a public repository, be mindful of the following security considerations:

1. **API Keys & Credentials**: Never commit API keys, tokens, or passwords to the repository.
   - Always use environment variables loaded from `.env` file (which is gitignored)
   - See `.env.example` for required variables without actual values

2. **Sensitive Memory Data**: The memory directory may contain sensitive information.
   - Specific sensitive files in memory/ are excluded in .gitignore
   - Review memory files before committing changes to ensure no sensitive data is included

3. **Rate Limiting**: The system includes API rate limiting to prevent accidental abuse of external services.
   - These limits can be adjusted in `core/api_manager.py`

## 📜 License

This project is licensed under Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) - see the LICENSE file for details. 