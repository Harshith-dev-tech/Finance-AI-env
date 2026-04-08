from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Finance AI Environment Running"}

@app.post("/reset")
def reset():
    return {
        "state": {
            "balance": 10000,
            "savings": 0,
            "spending": 0
        },
        "message": "Environment reset"
    }

@app.post("/step")
def step(action: dict):
    return {
        "state": {
            "balance": 9500,
            "savings": 500,
            "spending": 500
        },
        "reward": 1,
        "done": False
    }
