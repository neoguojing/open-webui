from langchain.prompts import MessagesPlaceholder,ChatPromptTemplate
from langchain_core.messages import HumanMessage, BaseMessage,SystemMessage,AIMessage


english_traslate_template = ChatPromptTemplate.from_messages(
    [
        HumanMessage(content="Translate the following into English and only return the translation result: {text}")
    ]
)

agent_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK, another assistant with different tools "
                " will help where you left off. Execute what you can to make progress."
                " If you or any of the other assistants have the final answer or deliverable,"
                " prefix your response with FINAL ANSWER so the team knows to stop."
                " You have access to the following tools: {tool_names}.\n{system_message}"),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )