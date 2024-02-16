from sentence_transformers import SentenceTransformer
import uuid


model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 사용자 질문 벡터화
def questrion_vectorize(question):
    vector = model.encode([question])[0].tolist()
    return vector

# 초기 데이터 벡터화
def vectorize_data(data):
    vectorized_list = [item['category'] for item in data]
    vectors = model.encode(vectorized_list)
    return vectors

# 직업별 임금정보 메타데이터 생성
def generate_metadata(data):
    metadata_list = [{
        "category": item['category'], 
        "id": str(uuid.uuid4()), 
        "total": item["metadata"]["total"],
        "bottom": item["metadata"]["bottom"],
        "middle": item["metadata"]["middle"],
        "top": item["metadata"]["top"],
    } for item in data]
    return metadata_list

# 업종별 직업정보 메타데이터 생성
def generate_sectors_metadata(data):
    metadata_list = [{
        "category": item['category'], 
        "id": str(uuid.uuid4()), 
        "jobs": item["jobs"],
    } for item in data]
    return metadata_list

# Pinecone에 업로드할 데이터 포맷
def format_vectors(vectors, metadata_list):
    formatted_vectors = [(metadata["id"], vector.tolist(), metadata) for vector, metadata in zip(vectors, metadata_list)]
    return formatted_vectors

# Pinecone에서 벡터 검색
def search_vector(index, query_vector):
    search_results = index.query(
        namespace="job1",
        vector=query_vector,
        top_k=5,
        include_values=True,
        include_metadata=True
    )
    return search_results