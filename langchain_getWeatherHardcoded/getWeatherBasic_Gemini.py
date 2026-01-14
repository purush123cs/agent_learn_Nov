import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables from .env file
load_dotenv()

@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

# Initialize the model with API key
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Create agent with tools (no prompt argument in this version)
agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt = SystemMessage(content="You are a helpful assistant."),
    debug=True
)

# Invoke by passing messages explicitly (required for your installed version)
result = agent.invoke({
    "messages": [
        HumanMessage(content="What is the weather in chennai?"),
    ]
})

# Print only the assistant's final text
output = None
if isinstance(result, dict):
    if "output" in result and result["output"]:
        output = result["output"]
    elif "messages" in result and result["messages"]:
        last_msg = result["messages"][-1]
        try:
            output = getattr(last_msg, "content", None) or str(last_msg)
        except Exception:
            output = str(last_msg)

if output is None:
    output = str(result)

print(output)