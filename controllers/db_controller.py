from fastapi import HTTPException, APIRouter,Body
from pinecone import Pinecone, PodSpec
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
from dotenv import load_dotenv
import os

db_router = APIRouter(tags=["Vecror DB 관리"])

MONGO_DETAILS = os.getenv("MONGO_DB_URL")
client = AsyncIOMotorClient(MONGO_DETAILS)
db = client.embedding_test
collection = db.model_comparison_results
load_dotenv()
# app = FastAPI()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
@db_router.get("/status/", description="생성된 Vector DB 조회")
async def db_status():
    try:
        if pc.list_indexes():
            dblist = pc.list_indexes()[0]
            index = pc.Index("test")
            vector_list = index.describe_index_stats()
            result = {
                "db_name": dblist["name"],
                "dimension": dblist["dimension"],
                "total_vectors": vector_list["total_vector_count"]
            }
            return result
        else:
            return {"message": "생성된 Vector DB가 없습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))    
    
@db_router.post("/delete/", description="Vector DB 삭제")
async def delete_db():
    try:
        if pc.list_indexes():
            pc.delete_index("test")
            return {"message": "Vector DB deleted successfully!"}
        else:
            return {"message": "삭제할 Vector DB가 없습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@db_router.post("/data_upload/", description="MongoDB에 데이터 업로드")
async def data_upload(data: dict = Body(...)):
    try:
        current_time = datetime.now().isoformat()
        data_with_time= {**data, "timestamp": current_time}
        # MongoDB에 데이터 삽입
        result = await collection.insert_one(data_with_time)
        if result.inserted_id:
            return {"message": "데이터 업로드 완료", "id": str(result.inserted_id)}
        else:
            return {"message": "데이터 업로드 실패"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@db_router.get("/data_retrieve/", description="MongoDB에 데이터 조회")
async def data_retrieve():
    try:
        documents = []
        cursor = collection.find({})
        async for document in cursor:
            document['_id'] = str(document['_id'])  # ObjectId를 문자열로 변환
            documents.append(document)
        return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))