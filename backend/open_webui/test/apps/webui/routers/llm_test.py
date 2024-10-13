import unittest


class TestLangchainApp(unittest.TestCase):

    def setUp(self):
        from open_webui.apps.lanchain.llm_app import LangchainApp  # 替换为实际模块名
        from open_webui.apps.lanchain.retriever import KnowledgeManager
        from open_webui.apps.lanchain.vectore_store import CollectionManager
        
        # 初始化 KnowledgeManager 和 LangchainApp
        self.nb = KnowledgeManager(data_path="/win/open-webui/backend/data/vector_db")
        # self.retrievers = None
        self.retrievers = self.nb.get_retriever(collection_names="194576726c78477d37df7fcad529e548a800e0e800b44d82f18b49df593b063",k=3)
        self.app = LangchainApp(retrievers=self.retrievers, db_path="sqlite:////win/open-webui/backend/data/langchain.db")
        self.collection_manager = CollectionManager("/win/open-webui/backend/data/vector_db")

    def test_ollama_response(self):
        
        print(self.collection_manager.list_collections())
        # self.nb.store(collection_name="aaaaa",source="/home/neo/Downloads/ir2023_ashare.docx",file_name="ir2023_ashare.docx")
        
        # resp = self.app("董事长报告书讲了什么？")
        # print("-----------------:",resp)
        stream_generator = self.app.ollama("董事长报告书讲了什么？")
        # 遍历生成器
        for response in stream_generator:
            print("iter:",response)
        # docs = nb.query_doc("194576726c78477d37df7fcad529e548a800e0e800b44d82f18b49df593b063","董事长报告书")
        # print(docs)
        # docs = self.app.query_doc("董事长报告书")
        # print("*****:\n",docs)
        # print(type(docs))
        
if __name__ == "__main__":
    unittest.main()
