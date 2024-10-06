from fastapi import FastAPI
from app.endpoints import router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.include_router(router)

origins = [
    "http://localhost:4200",  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Chat Dataset Analysis App!"}
