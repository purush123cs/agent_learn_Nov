from dotenv import load_dotenv
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

load_dotenv()

llm = init_chat_model("google_genai:gemini-2.5-flash-lite")

class State(MessagesState):
    message_type: str | None

graph_builder = StateGraph(State)

class MessageClassififer(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(
        ...,
        description="Classify if the message requires emotional (therapist) or logical response",
    )
     

def classifier(state: State):
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(MessageClassififer)
    result = classifier_llm.invoke([
        {
            "role": "system",
            "content": """Classify the user message as either:
            - 'emotional' : if it asks for emotional support, therapy, deals with feelings, or personal problems
            - 'logical' : if it asks for facts, information, logical analysis, or practical solutions
            """
        },
        {
            "role": "user",
            "content": last_message.content
        }
    ])
    return {"message_type": result.message_type}

def router(state: State):
    message_type = state.get("message_type", "logical")
    if message_type == "emotional":
        return {"next": "emotional"}
    
    return {"next": "logical"}

def emotional(state: State):
    last_message = state["messages"][-1]

    messages = [
        {
            "role": "system",
            "content": """You are an empathetic therapist. Provide emotional support and understanding in your responses.
                            Avoid giving logical solutions unless explicitly asked"""
        },
        {
            "role": "user",
            "content": last_message.content
        }
    ]

    response = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": response.content}]}

def logical(state: State):
    last_message = state["messages"][-1]

    messages = [
        {
            "role": "system",
            "content": """You are a logical and factual assistant. Provide clear, concise, and practical responses based on facts and logical reasoning.
                            Avoid emotional or empathetic language unless explicitly asked"""
        },
        {
            "role": "user",
            "content": last_message.content
        }
    ]

    response = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": response.content}]}


graph_builder.add_node("classifier", classifier)
graph_builder.add_node("router", router)
graph_builder.add_node("emotional", emotional)
graph_builder.add_node("logical", logical)
graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")
graph_builder.add_conditional_edges(
    "router",
    lambda state: state.get("next"),
        {"emotional": "emotional", "logical": "logical"}
)
graph_builder.add_edge("emotional", END)
graph_builder.add_edge("logical", END)

graph = graph_builder.compile()