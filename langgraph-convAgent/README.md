# LangGraph Conversational Agent

## Running the LangGraph Server

This uses uv, and not pip.
No need to install the packages separately. Just start the server.
To start the LangGraph development server, run:

```bash
uv run langgraph dev
```

## Input

```bash
postman request POST 'http://localhost:2024/runs/wait' \
  --header 'Content-Type: application/json' \
  --body '{
  "assistant_id": "agent",
  "input": {
    "messages": [
      {
        "role": "user",
        "content": "can you tell about Orcas in few sentences"
      }
    ]
  }
}'
```