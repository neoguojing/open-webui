import faiss
import numpy as np

class FaissDB:
    def __init__(self, dimension, nlist=100, nprobe=10):
        """
        初始化FaissDB类

        :param dimension: 向量的维度
        :param nlist: 聚类中心的数量，用于索引构建（用于 IVF 索引）
        :param nprobe: 检索时搜索的聚类中心数量，越大越准确但会降低性能
        """
        self.dimension = dimension
        self.nlist = nlist
        self.nprobe = nprobe
        self.index = None
        self.collections = {}
        self.counter = 0
    
    def create_collection(self, name):
        """
        创建一个新的向量集合
        
        :param name: 集合名称
        """
        if name in self.collections:
            raise ValueError(f"Collection {name} already exists")
        
        quantizer = faiss.IndexFlatL2(self.dimension)
        index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, faiss.METRIC_L2)
        self.collections[name] = {
            "index": index,
            "id_to_vector": {},
            "id_map": {}
        }
    
    def get_collection(self, name):
        """
        获取指定名称的集合
        
        :param name: 集合名称
        :return: 集合对象
        """
        return self.collections.get(name)
    
    def list_collections(self):
        """
        列出所有的集合名称
        
        :return: 集合名称列表
        """
        return list(self.collections.keys())
    
    def delete_collection(self, name):
        """
        删除指定的集合
        
        :param name: 集合名称
        """
        if name in self.collections:
            del self.collections[name]
        else:
            raise ValueError(f"Collection {name} does not exist")
    
    def train_collection(self, name, train_data):
        """
        训练集合中的索引
        
        :param name: 集合名称
        :param train_data: 用于训练的数据
        """
        collection = self.get_collection(name)
        if collection:
            collection["index"].train(train_data)
    
    def add(self, collection_name, vectors, ids=None):
        """
        添加向量到指定的集合中
        
        :param collection_name: 集合名称
        :param vectors: 向量数组
        :param ids: 可选的ID列表，如果不提供则自动生成
        """
        collection = self.get_collection(collection_name)
        if not collection:
            raise ValueError(f"Collection {collection_name} not found")
        
        if ids is None:
            ids = [self.counter + i for i in range(len(vectors))]
            self.counter += len(vectors)

        vectors = np.array(vectors).astype('float32')

        # 添加到集合
        for i, vector in zip(ids, vectors):
            collection["id_to_vector"][i] = vector
        
        # 添加到索引
        collection["index"].add(vectors)

        # 映射 ID 到索引中的位置
        collection["id_map"].update({i: len(collection["id_map"]) + j for j, i in enumerate(ids)})
    
    def upsert(self, collection_name, vectors, ids):
        """
        插入或更新向量到指定的集合中
        
        :param collection_name: 集合名称
        :param vectors: 向量数组
        :param ids: 要更新或插入的ID列表
        """
        self.add(collection_name, vectors, ids)  # upsert 在faiss中可以直接用add实现
    
    def query(self, collection_name, query_vector, k=5):
        """
        查询指定集合中最相似的向量
        
        :param collection_name: 集合名称
        :param query_vector: 查询向量
        :param k: 返回最近邻的数量
        :return: 最近邻的ID及其距离
        """
        collection = self.get_collection(collection_name)
        if not collection:
            raise ValueError(f"Collection {collection_name} not found")
        
        query_vector = np.array([query_vector]).astype('float32')
        distances, indices = collection["index"].search(query_vector, k)
        
        # 获取结果
        results = [(list(collection["id_map"].keys())[index], distances[0][i])
                   for i, index in enumerate(indices[0]) if index != -1]
        return results
    
    def get(self, collection_name, ids):
        """
        获取指定ID的向量
        
        :param collection_name: 集合名称
        :param ids: ID或ID列表
        :return: 向量或向量列表
        """
        collection = self.get_collection(collection_name)
        if not collection:
            raise ValueError(f"Collection {collection_name} not found")
        
        if isinstance(ids, list):
            return [collection["id_to_vector"].get(id_) for id_ in ids]
        else:
            return collection["id_to_vector"].get(ids)
    
    def delete(self, collection_name, ids):
        """
        删除指定ID的向量
        
        :param collection_name: 集合名称
        :param ids: ID或ID列表
        """
        collection = self.get_collection(collection_name)
        if not collection:
            raise ValueError(f"Collection {collection_name} not found")
        
        if isinstance(ids, list):
            for id_ in ids:
                if id_ in collection["id_to_vector"]:
                    del collection["id_to_vector"][id_]
                    del collection["id_map"][id_]
        else:
            if ids in collection["id_to_vector"]:
                del collection["id_to_vector"][ids]
                del collection["id_map"][ids]

    def set_nprobe(self, collection_name, nprobe):
        """
        设置nprobe的值，影响搜索的速度和准确性
        
        :param collection_name: 集合名称
        :param nprobe: 搜索的聚类中心数量
        """
        collection = self.get_collection(collection_name)
        if not collection:
            raise ValueError(f"Collection {collection_name} not found")
        
        collection["index"].nprobe = nprobe
