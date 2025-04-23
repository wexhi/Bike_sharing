"""Utility & helper functions."""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_community.chat_models.tongyi import ChatTongyi


def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    
    try:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        api = os.getenv("DASHSCOPE_API_KEY")
    except ValueError:     
        raise ValueError("Invalid API key provided.")
    
    if fully_specified_name == "qwen-plus" or fully_specified_name == "qwen-max":
        return ChatTongyi(model=fully_specified_name, api_key=api)
    
    provider, model = fully_specified_name.split("/", maxsplit=1)
    return init_chat_model(model, model_provider=provider)

