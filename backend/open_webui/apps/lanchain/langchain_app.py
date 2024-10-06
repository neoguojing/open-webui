from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.base import Runnable
from langchain.retrievers import EnsembleRetriever
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import OllamaEmbeddings


class LangchainApp:
    system_prompt = (
        "You are a helpful assistant. Answer all questions to the best of your ability."
        "Please use {language} as default language."
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
        "Please use {language} as default language."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    doc_qa_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
        "\n\n"
        "Please use {language} as default language."
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", doc_qa_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    llm: BaseChatModel
    runnable: Runnable
    with_message_history: RunnableWithMessageHistory
    db_path: str
    retrievers: EnsembleRetriever

    def __init__(self,model="qwen2.5:latest",db_path="sqlite:///memory.db",
                 retrievers=None,base_url="http://localhost:11434/v1/"):

        self.db_path = db_path
        self.llm =ChatOpenAI(
            model=model,
            # model="phi3.5:3.8b-mini-instruct-fp16",
            # model="llama3.1-local",
            openai_api_key="121212",
            base_url=base_url,
        )
        self.retrievers = retrievers
        if retrievers is not None:
            history_aware_retriever = create_history_aware_retriever(
                self.llm, self.retriever, self.contextualize_q_prompt
            )
            question_answer_chain = create_stuff_documents_chain(self.llm, self.qa_prompt)
            self.runnable = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        else:
            self.runnable = self.prompt | self.llm

        self.with_message_history = RunnableWithMessageHistory(
            self.runnable ,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            history_factory_config=[
                ConfigurableFieldSpec(
                    id="user_id",
                    annotation=str,
                    name="User ID",
                    description="Unique identifier for the user.",
                    default="",
                    is_shared=True,
                ),
                ConfigurableFieldSpec(
                    id="conversation_id",
                    annotation=str,
                    name="Conversation ID",
                    description="Unique identifier for the conversation.",
                    default="",
                    is_shared=True,
                ),
            ],
        )

    def get_session_history(self,user_id: str, conversation_id: str):
        return SQLChatMessageHistory(f"{user_id}--{conversation_id}", self.db_path)
    
    def chat(self,input: str,language="chinese",user_id="",conversation_id="",stream=True):
        if conversation_id == "":
            import uuid
            conversation_id = str(uuid.uuid4())

        input_template = {"language": language, "input": input}
        config = {"configurable": {"user_id": user_id, "conversation_id": conversation_id}}

        response = None
        if stream:
            response = self.with_message_history.stream(input_template,config)
            for item in response:
                yield item
        else:
            response = self.with_message_history.invoke(input_template,config)
            yield response
    
    def __call__(self,input: str,user_id="",conversation_id=""):
        response = self.chat(input=input,user_id=user_id,conversation_id=conversation_id)
        for item in response:
            # 从每个 item 中提取 'content'
            content = item.content
            # 使用 yield 生成提取的 content
            yield content

# if __name__ == "__main__":
#     app = LangchainApp()
#     # stream_generator = app.chat("介绍下南宋",stream=True)
#     # # 遍历生成器
#     # for response in stream_generator:
#     #     print(response.content)
#     ret = app.embed_query("我爱北京天安门")
#     print(ret)