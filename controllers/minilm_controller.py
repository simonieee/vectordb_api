from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter
from models.minilm_model import vectorize_data, generate_metadata, format_vectors, search_vector, questrion_vectorize
from pinecone import Pinecone, PodSpec
from dotenv import load_dotenv
import os
import json

minilm_router = APIRouter(tags=["paraphrase-multilingual-MiniLM-L12-v2 모델"])

load_dotenv()
# app = FastAPI()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY2"))

@minilm_router.post("/create_db/", description="Vector DB 생성 metric: Cosine 고정")
async def create_db():
    try:
        if pc.list_indexes():
            status = pc.list_indexes()[0]
            return {"message": "이미 생성된 Vector DB가 있습니다. 생성할 수 없습니다.", "db_name": status["name"], "dimension": status["dimension"]}
        pc.create_index("testtest", 384, metric="cosine",spec=PodSpec(environment="gcp-starter"))
        status = pc.list_indexes()[0]
        return {"message": "Vector DB created successfully!", "db_name": status["name"], "dimension": status["dimension"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@minilm_router.get("/status_db/", description="생성된 Vector DB 조회")
async def db_status():
    try:
        if pc.list_indexes():
            dblist = pc.list_indexes()[0]
            index = pc.Index("testtest")
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

@minilm_router.post("/delete_db/", description="Vector DB 삭제")
async def delete_db():
    try:
        if pc.list_indexes():
            pc.delete_index("testtest")
            return {"message": "Vector DB deleted successfully!"}
        else:
            return {"message": "삭제할 Vector DB가 없습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@minilm_router.post("/data_upload/", description="paraphrase-multilingual-MiniLM-L12-v2 모델 사용 Vector DB에 384차원 벡터 데이터 업로드")
async def upload_file(  file: UploadFile = File(...)):
    try:
        index = pc.Index("testtest")
        contents = await file.read()
        data = json.loads(contents)
        vectors = vectorize_data(data)
        metadata_list = generate_metadata(data)
        formatted_vectors = format_vectors(vectors, metadata_list)
        index.upsert(vectors=formatted_vectors, namespace="job2")
        return {"message": "Data uploaded successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@minilm_router.get("/search/", description="paraphrase-multilingual-MiniLM-L12-v2 모델 사용 질문에 대한 유사한 답변을 Vector DB에서 검색")
async def job_search( question: str):
    try:
        index = pc.Index("testtest")
        query_vector = questrion_vectorize(question)
        search_results = search_vector(index, query_vector)
        if not search_results.matches:
            return {"message": "No matches found"}
        d = [{"metadata": match["metadata"],
                "score": match["score"]}for match in search_results.matches]
        # result = {
        #     "metadata": search_results.matches[0]["metadata"],
        #     "score": search_results.matches[0]["score"]
        # }
        return d
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@minilm_router.post("/similarity_verification/", description="paraphrase-multilingual-MiniLM-L12-v2 모델 사용, 여러 질문에 대한 유사한 답변을 Vector DB에서 검색후 평균 스코어 및 결과 반환")
async def similarity_verification( file: UploadFile = File(...)):
    try:
        index = pc.Index("testtest")
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
            query_vector = questrion_vectorize(item)

            # Pinecone에서 벡터 검색
            search_result = index.query(
                namespace="job2",
                vector=query_vector,
                top_k=5,
                include_values=True,
                include_metadata=True
            )
            if search_result['matches']:
                result = {
                    "keyword": item,
                    "metadata": search_result['matches'][0]["metadata"],
                    "score": search_result['matches'][0]["score"]
                }
                if minimum_data["score"] > search_result['matches'][0]["score"]:
                    minimum_data["score"] = search_result['matches'][0]["score"]
                    minimum_data["keyword"] = item
                    minimum_data["metadata"] = search_result['matches'][0]["metadata"]
                search_results.append(result)
            else:
                search_results.append({"keyword": item, "metadata": {}, "score": 0.0})
        
        avg_score = sum([item["score"] for item in search_results]) / len(search_results) if search_results else 0
        req = {"data": search_results, "avg_score": avg_score, "minimum_data": minimum_data}
        return req
        return req
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
