from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from langgraph.graph import MessagesState

load_dotenv()

llm = init_chat_model("google_genai:gemini-2.5-flash-lite")

class State(MessagesState):
    output_message: str

class GraphOutput(TypedDict):
    output_message: str

graph_builder = StateGraph(State, output=GraphOutput)

def chatbot(state: State):
    #from langchain_core.messages import AIMessage
    #ai_message = AIMessage(content="Hello! This is a hardcoded response from the chatbot.")
    # This adds the message to the conversation history and sets output_message
    # return {
    #     "messages": [ai_message],
    #     "output_message": ai_message.content
    # }
    # Print all messages being sent to the LLM
    for msg in state["messages"]:
        print(f"{msg.__class__.__name__}: {msg.content}")
    llm_response = llm.invoke(state["messages"])
    return {"messages": [llm_response], "output_message": llm_response.content}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()