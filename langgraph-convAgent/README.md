# LangGraph Emotional/Logical Assistant

## Running the LangGraph Server

This uses uv, and not pip.

Create a .env file at the root. It should contain the api key. Ex:
```bash
GOOGLE_API_KEY="AI....d4"
LANGSMITH_API_KEY=ls...3a
```

No need to install the packages separately. Just start the server.
To start the LangGraph development server, run:

```bash
uv run langgraph dev
```

For VS code to resolve the imports of packages present in virtual env:\
It may be referring to some other virtual env. So, to make it refer to your virtual env, python interpreter needs to be set as below\
https://github.com/astral-sh/uv/issues/8706#issuecomment-2773598984\
Path ex: /Users/197002/Library/CloudStorage/OneDrive-Cognizant/Documents/purush_work/agent_learning_12_Dec_2025/lg-convAgent/.venv/bin/python3.13\
python3.13 was present under .venv/lib

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
        "content": "i am feeling nostalgic"
      }
    ]
  }
}'
```