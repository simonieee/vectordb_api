import chromadb
from chromadb.utils import embedding_functions
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import uuid
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class VectorDB:
    def __init__(self,db_name, model_name=None):
        self.db_name = db_name
        self.model_name = model_name
        if model_name:
          self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
        self.client = chromadb.PersistentClient(path="chroma")
        
    
    def create_db(self):
        if not self.model_name:
            return {"message": "모델 이름을 입력해주세요."}
        try:
            logging.info(f"-------------------Vector DB 생성중-------------------")
            self.client.get_or_create_collection(name=self.db_name,embedding_function=self.ef)
            logging.info(f"-------------------Vector DB 생성완료-------------------")
            return {"message": "Vector DB created successfully!"}
        except Exception as e:
            return {"message": str(e)}
        
    def data_upload(self,metadata_list):
        try:
            logging.info(f"-------------------데이터 업로드중-------------------")  
            collection = self.client.get_or_create_collection(name=self.db_name,embedding_function=self.ef)
            collection.upsert(documents=[item["category"] for item in metadata_list], ids=[item["id"] for item in metadata_list], metadatas=metadata_list)
            logging.info(f"-------------------데이터 업로드완료-------------------")
            return {"message": "데이터 업로드 완료"}
        except Exception as e:
            return {"message": str(e)}
        
    def db_status(self):
        try:
            logging.info(f"-------------------db_status 조회중-------------------")
            collection = self.client.get_or_create_collection(name=self.db_name)
            list = self.client.list_collections()
            logging.info(f"-------------------db_status 조회완료-------------------")
            return {"db_name": collection.name,"count":collection.count(),"list":list}
        except Exception as e:
            return {"message": str(e)}
    
    def generate_metadata(self, data):
        logging.info(f"-------------------metadata_list 생성중-------------------")
        metadata_list = [{
        "category": item['category'], 
        "id": str(uuid.uuid4()), 
        "total": item["metadata"]["total"],
        "bottom": item["metadata"]["bottom"],
        "middle": item["metadata"]["middle"],
        "top": item["metadata"]["top"],
        } for item in data]
        logging.info(f"-------------------metadata_list 생성완료-------------------")
        return metadata_list
    
    def delete_db(self):
        try:
            logging.info(f"-------------------{self.db_name} 삭제중-------------------")
            self.client.delete_collection(name=self.db_name)
            logging.info(f"-------------------{self.db_name} 삭제되었습니다-------------------")
            return {"message": f"VectorDB:{self.db_name}가 삭제되었습니다"}
        except Exception as e:
            return {"message": str(e)}

    def cosine_similarity_pytorch(self,embedding1, embedding2):
        # 두 임베딩을 텐서로 변환합니다.
        embedding1 = torch.tensor(embedding1)
        embedding2 = torch.tensor(embedding2)
        
        # F.cosine_similarity 함수를 사용하여 코사인 유사도를 계산합니다.
        # dim=0으로 설정하여 벡터 간 비교를 수행합니다.
        similarity = F.cosine_similarity(embedding1, embedding2, dim=0)
        
        return similarity.item()  # 텐서에서 Python 숫자로 변환

    def similarity(self, array, text_embedding):
        similarity_data = []
        for item in array:
          text2_embedding = self.ef(item)
          s = self.cosine_similarity_pytorch(text_embedding, text2_embedding[0])
          similarity_data.append(s)
        return similarity_data
        
    def search_text(self, text):
        try:
            logging.info(f"-------------------유사 직무정보 검색중-------------------")
            collection = self.client.get_or_create_collection(name=self.db_name, embedding_function=self.ef)
            search_data = collection.query(query_texts=[text], n_results=5, include=["metadatas","embeddings","documents"])
            smilarity_score = self.similarity(search_data["documents"][0],self.ef(text)[0])
            d = [{"metadata":item, "score":score} for item,score in zip(search_data["metadatas"][0],smilarity_score)]
            logging.info(f"-------------------검색완료-------------------")
            return d
        except Exception as e:
            return {"message": str(e)}