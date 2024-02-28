from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter
from models.vector_db import VectorDB
import json

vector_db_router = APIRouter(tags=["Chroma Vector DB 관리"])

@vector_db_router.post("/create_db/", description="Vector DB 생성")
async def create_db(model_name: str, db_name: str):
    try:
        vector_db = VectorDB(db_name=db_name, model_name=model_name)
        result = vector_db.create_db()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@vector_db_router.get("/status_db/", description="생성된 Vector DB 조회")
async def db_status(db_name: str):
    try:
        vector_db = VectorDB( db_name=db_name)
        result =  vector_db.db_status()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@vector_db_router.post("/data_upload/", description="Vector DB에 데이터 업로드")
async def data_upload(db_name: str,model_name:str, file: UploadFile = File(...)):
    try:
        vector_db = VectorDB(db_name=db_name, model_name=model_name)
        contents = await file.read()
        data = json.loads(contents)
        metadata_list = vector_db.generate_metadata(data)
        result = vector_db.data_upload(metadata_list=metadata_list)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@vector_db_router.post("/delete_db/", description="Vector DB 삭제")
async def delete_db(db_name: str):
    try:
        vector_db = VectorDB(db_name=db_name)
        result = vector_db.delete_db()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@vector_db_router.get("/search_vector/", description="Vector DB에서 벡터 검색")
async def search_vector(db_name: str, model_name: str,text: str):
    try:
        vector_db = VectorDB(db_name=db_name, model_name=model_name)
        result = vector_db.search_text(text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))