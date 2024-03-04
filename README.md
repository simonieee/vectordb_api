# ChromaDB + 허킹페이스 기반 임베딩모델 테스트베드

## Installation

```bash
$ pip install -r requirements.txt
```

## Running the app

```bash
# development
# worker는 알아서 지정
$ uvicorn main:app --reload --host=0.0.0.0 --port=8000
```

## ENV

```bash
# MongoDB URL
MONGO_DB_URL=
```
