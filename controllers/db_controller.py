from fastapi import HTTPException, APIRouter
from pinecone import Pinecone, PodSpec
from dotenv import load_dotenv
import os

db_router = APIRouter(tags=["Vecror DB 관리"])

load_dotenv()
# app = FastAPI()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

@db_router.post("/create_db/", description="Vector DB 생성 metric: Cosine 고정")
async def create_db(dbname: str, dimension: int):
    try:
        if pc.list_indexes():
            status = pc.list_indexes()[0]
            return {"message": "이미 생성된 Vector DB가 있습니다. 생성할 수 없습니다.", "db_name": status["name"], "dimension": status["dimension"]}
        pc.create_index(dbname, dimension, metric="cosine",spec=PodSpec(environment="gcp-starter"))
        return {"message": "Vector DB created successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@db_router.get("/db_status/", description="생성된 Vector DB 조회")
async def db_status():
    try:
        if pc.list_indexes():
            dblist = pc.list_indexes()[0]
            return {"db_name": dblist["name"], "dimension": dblist["dimension"]}
        else:
            return {"message": "생성된 Vector DB가 없습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))    
    
@db_router.post("/delete_db/", description="Vector DB 삭제")
async def delete_db(dbname: str):
    try:
        if pc.list_indexes():
            pc.delete_index(dbname)
            return {"message": "Vector DB deleted successfully!"}
        else:
            return {"message": "삭제할 Vector DB가 없습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@db_router.get("/db_dimensions/", description="현재 설정된 Vector DB dimensions 조회")
async def get_db_dimensions():
    try:
        if pc.list_indexes():
            dimension = pc.list_indexes()[0]["dimension"]
            return {"dimension": dimension}
        else :
            return {"message":"생성된 Vector DB가 없습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))