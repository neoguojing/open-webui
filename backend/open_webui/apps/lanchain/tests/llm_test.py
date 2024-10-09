import unittest
from open_webui.apps.lanchain.llm_app import LangchainApp  # 替换为实际模块名
from open_webui.apps.lanchain.retriever import KnowledgeManager

class TestLangchainApp(unittest.TestCase):

    def setUp(self):
        # 初始化 KnowledgeManager 和 LangchainApp
        self.nb = KnowledgeManager(data_path="/win/open-webui/backend/data/vector_db")
        self.retrievers = self.nb.get_retriever("194576726c78477d37df7fcad529e548a800e0e800b44d82f18b49df593b063")
        self.app = LangchainApp(retrievers=self.retrievers, db_path="sqlite:////win/open-webui/backend/data/langchain.db")

    def test_ollama_response(self):
        # 测试 ollama 方法的响应
        stream_generator = self.app.ollama("董事长报告书讲了什么？")
        responses = list(stream_generator)
        for response in responses:
            print(response)
        self.assertGreater(len(responses), 0, "Expected responses from the generator")
        
if __name__ == "__main__":
    unittest.main()
