from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, UploadFile, File,HTTPException
from pinecone import Pinecone
from dotenv import load_dotenv
import uuid
import json
import os

load_dotenv()
app = FastAPI()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("test")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # 파일 내용을 읽고 JSON으로 파싱
        contents = await file.read()
        data = json.loads(contents)
        
        # 데이터 처리 및 벡터화
        vectorized_list = [item['category'] for item in data]
        vectors = model.encode(vectorized_list)
        
        # 메타데이터 생성
        # vector db에 저장할 때 metadata는 Dictionary 형태로 1:1 key-value 형식을 맞춰줘야함
        metadata_list = [{
            "category": item['category'], 
            "id": str(uuid.uuid4()), 
            "total": item["metadata"]["total"],
            "bottom": item["metadata"]["bottom"],
            "middle": item["metadata"]["middle"],
            "top": item["metadata"]["top"],
        } for item in data]
        
        # 벡터와 메타데이터 포매팅
        formatted_vectors = [(metadata["id"], vector.tolist(), metadata) for vector, metadata in zip(vectors, metadata_list)]
  
        # # Pinecone에 벡터 데이터 저장
        index.upsert(vectors=formatted_vectors, namespace="job3")
        
        return {"message": "Data uploaded successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobsearch/")
async def job_search(question: str):
    try:
        # 검색어 벡터화
        query_vector = model.encode([question])[0].tolist()

        # Pinecone에서 벡터 검색
        search_results = index.query(
            namespace="job3",
            vector=query_vector,
            top_k=5,
            include_values=True,
            include_metadata=True
        )

        # 검색 결과 리턴
        if not search_results.matches:
            return {"message": "No matches found"}
        
        result = {
            "metadata": search_results.matches[0]["metadata"],
            "score": search_results.matches[0]["score"]
        }
        return result
        # return query_vector
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/similarity_verification/")
async def similarity_verification(file: UploadFile = File(...)):
    try:
        content = await file.read()
        data = json.loads(content)
        search_results = []
        for item in data:
            # 검색어 벡터화
            query_vector = model.encode([item])[0].tolist()

            # Pinecone에서 벡터 검색
            search_result = index.query(
                namespace="job3",
                vector=query_vector,
                top_k=5,
                include_values=True,
                include_metadata=True
            )
            if search_result.matches:
                result = {
                    "keyword": item,
                    "metadata": search_result.matches[0]["metadata"],
                    "score": search_result.matches[0]["score"]
                }
                search_results.append(result)
            else:
                search_results.append({"keyword":item, "metadata":{}, "score": 0.0})
           
        return search_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)