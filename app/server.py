from fastapi import FastAPI
from food_agent import food_agent
from langchain.pydantic_v1 import BaseModel
from langserve import add_routes

app = FastAPI(
    title="Sinocare Image Agent",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)


class Input(BaseModel):
    input: str


add_routes(
    app,
    food_agent().with_types(input_type=Input),
    path="/food",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
