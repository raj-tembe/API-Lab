# Flask-API

Simple collection of example Flask endpoints demonstrating chatbot and agent workflows.

## Files
- `basic_chatbot_api.py`: Minimal Flask `/api/chat` endpoint that forwards `query` to `llmchain.chain`.
- `simple_LangGraph_chatbot_api.py`: Uses a `workflow.graph` (LangGraph) to process queries at `/api/chat`.
- `simple_RAG_chatbot_api.py`: Retrieval-Augmented Generation example; retrieves docs from `vectorstore.db`, reranks with `CrossEncoder`, then calls `llmchain.chain`.
- `Human_In_the_Loop_api.py`: LangGraph workflow example that may interrupt for human input. Endpoints:
  - `POST /api/chat` — start workflow (may return `waiting_for_human`).
  - `POST /api/human-response` — resume workflow with `human_response`.
- `Supervise_MultiAI_Agent_API.py`: Example multi-agent supervisor that invokes `multi_agent.graph` with a `HumanMessage` and returns structured results.
- `LICENSE`: Repository license.

## Quick start

1. Install Python dependencies used by the examples (adjust as needed):

   ```
   pip install flask sentence-transformers
   ```

   Note: Some modules in these examples (`llmchain`, `workflow`, `langgraph`, `multi_agent`, `vectorstore`, `langchain_core`) are local and project-specific. Ensure those packages or modules are available in project directory.

2. Run an API script (choose one):

   ```
   python basic_chatbot_api.py
   ```

3. Call the endpoint with JSON (example):

   ```
   curl -X POST http://localhost:5000/api/chat \
     -H "Content-Type: application/json" \
     -d '{"query": "Hello, how are you?"}'
   ```

4. For `Human_In_the_Loop_api.py`, if the `/api/chat` response indicates `waiting_for_human`, call:

   ```
   curl -X POST http://localhost:5000/api/human-response \
     -H "Content-Type: application/json" \
     -d '{"thread_id":"1","human_response":"Your input here"}'
   ```

## Notes
- These scripts are simple examples and assume supporting modules and data (e.g., vector store, models) are configured.
- Adjust ports or debug settings as needed. Several scripts use `port=5000` explicitly.
- Feel free to open issues or improve the repo with a `requirements.txt` or setup instructions.
