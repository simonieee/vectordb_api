from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter
from models.large_model import load_data, vectorize_texts, search_vector,vectorize_question
from models.minilm_model import generate_metadata, format_vectors
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import json

large_router = APIRouter(tags=["intfloat/multilingual-e5-large 모델"])

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# index = pc.Index("test")

@large_router.post("/large_model_upload/")
async def large_upload_file(dbname:str,file: UploadFile = File(...)):
    try:
        index = pc.Index(dbname)
        contents = await file.read()
        data = json.loads(contents)
        vectors = vectorize_texts(data)
        metadata_list = generate_metadata(data)
        formatted_vectors = format_vectors(vectors, metadata_list)
        index.upsert(vectors=formatted_vectors, namespace="job1")
        return {"message": "Data uploaded successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@large_router.get("/large_model_search/")
async def large_search_vector(dbname:str, question: str):
    try:
        index = pc.Index(dbname)
        vector = vectorize_question(question)[0].tolist()
        search_results = search_vector(index, vector)
        if not search_results.matches:
            return {"message": "No matches found"}
        result = {
            "metadata": search_results.matches[0]["metadata"],
            "score": search_results.matches[0]["score"]
        }
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@large_router.post("/large_model_similarity_verification/")
async def large_similarity_verification(dbname:str,file: UploadFile = File(...)):
    try:
        index = pc.Index(dbname)
        content = await file.read()
        data = json.loads(content)
        search_results = []
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
                search_results.append(search_data)
            req = {"data":search_results, "avg_score": sum([item["score"] for item in search_results]) / len(search_results)}
        return req
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))