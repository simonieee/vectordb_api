from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter
from models.large_model import load_data, vectorize_texts, search_vector,vectorize_question,search_sector_vector
from models.minilm_model import generate_metadata, format_vectors, generate_sectors_metadata
from pinecone import Pinecone,PodSpec
from dotenv import load_dotenv
import os
import json

large_router = APIRouter(tags=["intfloat/multilingual-e5-large 모델"])

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# index = pc.Index("test")

@large_router.post("/create_db/", description="Vector DB 생성 metric: Cosine 고정")
async def create_db():
    try:
        if pc.list_indexes():
            status = pc.list_indexes()[0]
            return {"message": "이미 생성된 Vector DB가 있습니다. 생성할 수 없습니다.", "db_name": status["name"], "dimension": status["dimension"]}
        pc.create_index("test", 1024, metric="cosine",spec=PodSpec(environment="gcp-starter"))
        status = pc.list_indexes()[0]
        return {"message": "Vector DB created successfully!","db_name": status["name"], "dimension": status["dimension"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@large_router.get("/status_db/", description="생성된 Vector DB 조회")
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

@large_router.post("/delete_db/", description="Vector DB 삭제")
async def delete_db():
    try:
        if pc.list_indexes():
            pc.delete_index("test")
            return {"message": "Vector DB deleted successfully!"}
        else:
            return {"message": "삭제할 Vector DB가 없습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@large_router.post("/data_upload/", description="Vector DB에 데이터 업로드")
async def large_upload_file(file: UploadFile = File(...)):
    try:
        index = pc.Index("test")
        contents = await file.read()
        data = json.loads(contents)
        vectors = vectorize_texts(data)
        metadata_list = generate_metadata(data)
        formatted_vectors = format_vectors(vectors, metadata_list)
        index.upsert(vectors=formatted_vectors, namespace="job1")
        return {"message": "Data uploaded successfully!", "result":True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
      
@large_router.post("/sectors_upload/", description="Vector DB에 분야별 직업정보 데이터 업로드")
async def large_sectors_upload_file(dbname:str,file: UploadFile = File(...)):
    try:
        index = pc.Index(dbname)
        contents = await file.read()
        data = json.loads(contents)
        vectors = vectorize_texts(data)
        metadata_list = generate_sectors_metadata(data)
        formatted_vectors = format_vectors(vectors, metadata_list)
        index.upsert(vectors=formatted_vectors, namespace="sector1")
        return {"message": "Data uploaded successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))    
    
@large_router.get("/search/")
async def large_search_vector(question: str):
    try:
        index = pc.Index("test")
        vector = vectorize_question(question)[0].tolist()
        search_results = search_vector(index, vector)
        if not search_results.matches:
            return {"message": "No matches found"}
        d = [{"metadata": match["metadata"],
                "score": match["score"]}for match in search_results.matches]
        return d
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@large_router.get("/sector_search/")
async def large_sector_search_vector(dbname:str, question: str):
    try:
        index = pc.Index(dbname)
        vector = vectorize_question(question)[0].tolist()
        search_results = search_sector_vector(index, vector)
        if not search_results.matches:
            return {"message": "No matches found"}
        d = [{"metadata": match["metadata"],
                "score": match["score"]}for match in search_results.matches]
        return d
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# 질문 여러개에 대한 유사한 답변을 Vector DB에서 검색후 평균 스코어 및 결과 반환
@large_router.post("/similarity_verification/")
async def large_similarity_verification(file: UploadFile = File(...)):
    try:
        index = pc.Index("test")
        content = await file.read()
        data = json.loads(content)
        search_results = []
        minimum_data ={
            "keyword": "",
            "metadata": "",
            "score": 1.0
        }
        for item in data:
            # 검색어 벡터화
            vector = vectorize_question(item)[0].tolist()
            result = search_vector(index, vector)
            if result.matches:
                search_data = {
                    "keyword": item,
                    "metadata": result.matches[0]["metadata"],
                    "score": result.matches[0]["score"]
                }
                if(minimum_data["score"]>result.matches[0]["score"]):
                    minimum_data["score"] = result.matches[0]["score"]
                    minimum_data["keyword"] = item
                    minimum_data["metadata"] = result.matches[0]["metadata"]
                search_results.append(search_data)
            req = {"data":search_results, "avg_score": sum([item["score"] for item in search_results]) / len(search_results), "minimum_data": minimum_data}
        return req
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    