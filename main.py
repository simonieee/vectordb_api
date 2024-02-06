from fastapi import FastAPI
from controllers.minilm_controller import minilm_router
from controllers.large_controller import large_router 
from controllers.db_controller import db_router
app = FastAPI()

# FastAPI 앱에 컨트롤러 라우트 추가 (모델별로 분류)
app.include_router(minilm_router)
app.include_router(large_router)
app.include_router(db_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)