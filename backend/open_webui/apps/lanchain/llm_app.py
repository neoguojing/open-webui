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
from langchain_ollama.llms import OllamaLLM
from langchain_core.messages.ai import AIMessage,AIMessageChunk
from langchain_core.runnables.utils import AddableDict
from prompt import default_template,contextualize_q_template,doc_qa_template
import json
from datetime import datetime,timezone

class LangchainApp:
    
    llm: BaseChatModel
    runnable: Runnable
    with_message_history: RunnableWithMessageHistory
    db_path: str
    retrievers: EnsembleRetriever

    def __init__(self,model="qwen2.5:14b",db_path="sqlite:///langchain.db",
                 retrievers=None,base_url="http://localhost:11434/v1/"):

        self.db_path = db_path
        self.model = model
        self.llm =ChatOpenAI(
            model=model,
            # model="phi3.5:3.8b-mini-instruct-fp16",
            # model="llama3.1-local",
            openai_api_key="121212",
            base_url=base_url,
        )
        
        # self.llm  = OllamaLLM(model=model,base_url="http://localhost:11434")
        self.retrievers = retrievers
        if retrievers is not None:
            history_aware_retriever = create_history_aware_retriever(
                self.llm, self.retrievers, contextualize_q_template
            )
            question_answer_chain = create_stuff_documents_chain(self.llm, doc_qa_template)
            self.runnable = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        else:
            self.runnable = default_template | self.llm 

        self.with_message_history = RunnableWithMessageHistory(
            self.runnable ,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
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
                )
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
            return response
        else:
            response = self.with_message_history.invoke(input_template,config)
            return response
    
    def ollama(self,input: str,user_id="",conversation_id="",stream=True):
        response = self.chat(input=input,user_id=user_id,conversation_id=conversation_id,stream=stream)
        content = None
        if not stream:
            print(response,type(response))
            utc_now = datetime.now(timezone.utc)
            utc_now_str = utc_now.isoformat() + 'Z'
            if isinstance(response,AIMessage):
                content = response.content
            elif isinstance(response,dict):
                content = response['answer']
                
            print("----------",content)
            message_data = {
                "model": self.model,
                "created_at": utc_now_str,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "done": True,
            }
            return message_data
        else:
            is_done = False
            finish_reason = None
            for item in response:
                print(item,type(item))
                # 从每个 item 中提取 'content'
                if isinstance(item,AIMessageChunk):
                    content = item.content
                    if item.response_metadata:
                        is_done = True
                        finish_reason = item.response_metadata['finish_reason']
                elif isinstance(item,AddableDict):
                    content = item.get('answer')
                    if content is None:
                        is_done = True
                
                
                utc_now = datetime.now(timezone.utc)
                utc_now_str = utc_now.isoformat() + 'Z'
                message_data = {
                    "model": self.model,
                    "created_at": utc_now_str,
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "done": is_done,
                    "done_reason": finish_reason
                }
                
                yield json.dumps(message_data) + "\n"  # 添加换行符
    
    def __call__(self,input: str,user_id="",conversation_id=""):
        response = self.chat(input=input,user_id=user_id,conversation_id=conversation_id)
        for item in response:
            # 从每个 item 中提取 'content'
            content = item.content
            # 使用 yield 生成提取的 content
            yield content

# if __name__ == "__main__":
#     app = LangchainApp()
#     stream_generator = app.ollama("hello")
#     # 遍历生成器
#     for response in stream_generator:
#         print(response)
#     # ret = app.embed_query("我爱北京天安门")
#     # print(ret)