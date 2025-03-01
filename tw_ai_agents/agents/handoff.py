"""Handoff functionality for supervisor agent system.

This module provides tools for transferring control between agents in a multi-agent system.
"""

import re
import uuid

from langchain_core.messages import AIMessage, ToolMessage, ToolCall
from langchain_core.tools import tool, BaseTool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from pydantic import BaseModel
from typing_extensions import Annotated


WHITESPACE_RE = re.compile(r"\s+")


def _normalize_agent_name(agent_name: str) -> str:
    """Normalize an agent name to be used inside the tool name.

    Args:
        agent_name: The name of the agent to normalize.

    Returns:
        A normalized version of the agent name.
    """
    return WHITESPACE_RE.sub("_", agent_name.strip()).lower()


def create_handoff_tool(
    *, agent_name: str, agent_description: str = ""
) -> BaseTool:
    """Create a tool that can handoff control to the requested agent.

    Args:
        agent_name: The name of the agent to handoff control to, i.e.
            the name of the agent node in the multi-agent graph.
            Agent names should be simple, clear and unique, preferably in snake_case,
            although you are only limited to the names accepted by LangGraph
            nodes as well as the tool names accepted by LLM providers
            (the tool name will look like this: `transfer_to_<agent_name>`).

    Returns:
        A tool that can be used to transfer control to another agent.
    """
    tool_name = f"transfer_to_{_normalize_agent_name(agent_name)}"

    class BaseArgsSchema(BaseModel):
        tool_call_id: Annotated[str, InjectedToolCallId]
        message_for_subagent: str

    @tool(tool_name, description=agent_description, args_schema=BaseArgsSchema)
    def handoff_to_agent(
        tool_call_id: Annotated[str, InjectedToolCallId],
        message_for_subagent: str,
    ) -> Command:
        """Ask another agent for help."""
        tool_message = ToolMessage(
            content=f"Successfully transferred to {agent_name}\n\n"
            f"## Message from the supervisor\n"
            f"{message_for_subagent}",
            name=tool_name,
            tool_call_id=tool_call_id,
        )
        return Command(
            goto=agent_name,
            graph=Command.PARENT,
            update={"messages": [tool_message]},
        )

    return handoff_to_agent


def create_handoff_back_messages(
    agent_name: str, supervisor_name: str
) -> tuple[AIMessage, ToolMessage]:
    """Create a pair of (AIMessage, ToolMessage) to add to the message history when returning control to the supervisor."""
    tool_call_id = str(uuid.uuid4())
    tool_name = f"transfer_back_to_{_normalize_agent_name(supervisor_name)}"
    tool_calls = [ToolCall(name=tool_name, args={}, id=tool_call_id)]
    return (
        AIMessage(
            content=f"Transferring back to {supervisor_name}",
            tool_calls=tool_calls,
            name=agent_name,
        ),
        ToolMessage(
            content=f"Successfully transferred back to {supervisor_name}",
            name=tool_name,
            tool_call_id=tool_call_id,
        ),
    )
