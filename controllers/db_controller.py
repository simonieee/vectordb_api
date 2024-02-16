from fastapi import HTTPException, APIRouter
from pinecone import Pinecone, PodSpec
from dotenv import load_dotenv
import os

db_router = APIRouter(tags=["Vecror DB 관리"])

load_dotenv()
# app = FastAPI()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
@db_router.get("/status/", description="생성된 Vector DB 조회")
async def db_status():
    try:
        if pc.list_indexes():
            dblist = pc.list_indexes()[0]
            return {"db_name": dblist["name"], "dimension": dblist["dimension"]}
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
