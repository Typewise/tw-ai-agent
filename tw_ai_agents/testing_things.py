import time

from tw_ai_agents.agents.graph_creator import (
    get_complete_graph,
    get_input_configs,
)
from tw_ai_agents.agents.llm_models_loader import get_llm_model
from tw_ai_agents.config_handler.pydantic_models.agent_models import (
    AgentResponseRequest,
)
from tw_ai_agents.server import process_agent_response

input_configs = get_input_configs()
model_name: str = "openai/gpt-4o"
model = get_llm_model(model_name)
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
supervisor = get_complete_graph(
    model,
    input_configs,
    memory=memory,
    channel_type_id="67bed9fe3b2f84a3a5e67779",
)
full_compiled_graph = supervisor.get_supervisor_compiled_graph()
# full_compiled_graph.get_graph().draw_mermaid_png(output_file_path="graph.png")


def main():
    discussion_id = f"123456_{int(time.time())}"
    input_data = AgentResponseRequest(
        message_type="user",
        # message_text="Typewise sucks.",
        message_text="Hello!",
        discussion_id=discussion_id,
        client="typewise",
        channel_type_id="67bed9fe3b2f84a3a5e67779",
    )
    output = process_agent_response(input_data)
    print(output)
    input_data = AgentResponseRequest(
        message_type="user",
        # message_text="Typewise sucks.",
        message_text="Hey. All good? I'd like that to update my address information in my account.",
        discussion_id=discussion_id,
        client="typewise",
        channel_type_id="67bed9fe3b2f84a3a5e67779",
    )
    output = process_agent_response(input_data)
    print(output)

    second_input_data = AgentResponseRequest(
        message_type="agent",
        # message_type="user",
        message_text="My email is f.roberts@gmail.com",
        discussion_id=discussion_id,
        client="typewise",
        channel_type_id="67bed9fe3b2f84a3a5e67779",
    )
    second_output = process_agent_response(second_input_data)
    print(second_output)

    third_input_data = AgentResponseRequest(
        message_type="agent",
        # message_type="user",
        message_text="Please update to Bohstrasse 322, Bern, Switzerland.",
        discussion_id=discussion_id,
        client="typewise",
        channel_type_id="67bed9fe3b2f84a3a5e67779",
    )
    third_output = process_agent_response(third_input_data)
    print(third_output)

    a = 1


if __name__ == "__main__":
    main()
