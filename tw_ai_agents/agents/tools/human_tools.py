from langchain_core.tools import tool
from langgraph.types import interrupt

COMPLETE_HANDOFF_STRING = "Handoff the full conversation to a real agent."
ASK_USER_TOOL_NAME = "ask_to_user"


@tool("real_human_agent_execute_actions")
def real_human_agent_execute_actions(query: str) -> str:
    """
    Make a real agent execute actions.
    This tool makes a real human agent execute actions based on the provided query.

    Args:
        query: The query to make the real human agent execute.

    Returns:
        A string containing the information from the real human agent. It can be a confirmation or a negative answer.
    """
    a = 1

    answer = interrupt(
        # This value will be sent to the client
        # as part of the interrupt information.
        query,
    )
    print(f"> Received an input from the interrupt: {answer}")
    return answer


@tool("handoff_conversation_to_real_agent")
def handoff_conversation_to_real_agent() -> str:
    """
    Handoff the full conversation to a real agent.
    """
    answer = interrupt(
        # This value will be sent to the client
        # as part of the interrupt information.
        COMPLETE_HANDOFF_STRING,
    )
    return answer


def get_ask_to_user_tool(prepare_question_func):
    @tool(ASK_USER_TOOL_NAME, parse_docstring=True)
    def ask_to_user(query_for_the_user: str) -> str:
        """
        Ask a question to the end-user.
        This tool must be used when some refinement is needed in the information provided by the end-user to better solve the task.

        Args:
            query_for_the_user: The query to ask the end-user.

        Returns:
            The answer from the end-user.
        """
        question = prepare_question_func(query_for_the_user)
        answer = interrupt(
            # This value will be sent to the client
            # as part of the interrupt information.
            question,
        )
        return answer

    return ask_to_user
