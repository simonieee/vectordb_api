from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import json

# 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')

# 초기 데이터파일 로드
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# 시퀀스에 대한 고정된 길이의 벡터 생성(1024차원)
def average_pooling(last_hidden_states, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

# 데이터 벡터화
def vectorize_texts(data):
    # 초기 데이터파일에서 카테고리 데이터 추출
    texts = [item['category'] for item in data]
    # 토크나이저로 토큰화
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    # 모델로 벡터화 ** 키-값 쌍으로 model에 입력
    outputs = model(**inputs)
    pooled_output = average_pooling(outputs.last_hidden_state, inputs['attention_mask'])
    return pooled_output


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

def vectorize_question(question):
    # 텍스트를 모델 입력 형식으로 토크나이즈
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # 모델을 사용하여 토크나이즈된 입력의 출력 계산
    outputs = model(**inputs)
    
    # 출력의 마지막 은닉 상태를 가져옴
    last_hidden_states = outputs.last_hidden_state
    
    # 주의 마스크를 적용하여 토큰들의 평균 계산
    input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden_states.size()).float()
    sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask
    
    return mean_embeddings