from fastapi import FastAPI
from controllers.minilm_controller import minilm_router
from controllers.large_controller import large_router 
from controllers.db_controller import db_router
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 오리진 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 메소드 허용 (GET, POST, DELETE, etc.)
    allow_headers=["*"],  # 모든 헤더 허용
)
# FastAPI 앱에 컨트롤러 라우트 추가 (모델별로 분류)
app.include_router(minilm_router, prefix="/minilm_model")
app.include_router(large_router, prefix="/large_model")
app.include_router(db_router, prefix="/db")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    