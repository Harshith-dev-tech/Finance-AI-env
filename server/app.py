from fastapi import FastAPI, Request, HTTPException
from typing import Dict
import uuid

from env import PersonalFinanceEnv, Action

app = FastAPI()

# ─────────────────────────────────────────────
# SESSION STORAGE (multi-user support)
# ─────────────────────────────────────────────

sessions: Dict[str, PersonalFinanceEnv] = {}
observations: Dict[str, dict] = {}


# ─────────────────────────────────────────────
# ROOT
# ─────────────────────────────────────────────

@app.get("/")
def home():
    return {"message": "Finance AI Environment Running"}


# ─────────────────────────────────────────────
# RESET (Create new environment session)
# ─────────────────────────────────────────────

@app.post("/reset")
def reset():
    session_id = str(uuid.uuid4())

    env = PersonalFinanceEnv()
    obs = env.reset()

    sessions[session_id] = env
    observations[session_id] = obs

    return {
        "session_id": session_id,
        "state": obs.dict(),
        "message": "Environment reset"
    }


# ─────────────────────────────────────────────
# STEP (Core RL interaction)
# ─────────────────────────────────────────────

@app.post("/step")
async def step(request: Request):
    data = await request.json()

    session_id = data.get("session_id")
    action_data = data.get("action")

    if not session_id or session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid or missing session_id")

    if not action_data:
        raise HTTPException(status_code=400, detail="Missing action")

    env = sessions[session_id]

    try:
        action = Action(**action_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid action format: {str(e)}")

    obs, reward, done, info = env.step(action)

    observations[session_id] = obs

    return {
        "state": obs.dict(),
        "reward": reward.dict(),
        "done": done,
        "info": info
    }


# ─────────────────────────────────────────────
# OPTIONAL: GET CURRENT STATE
# ─────────────────────────────────────────────

@app.get("/state/{session_id}")
def get_state(session_id: str):
    if session_id not in observations:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "state": observations[session_id].dict()
    }


# ─────────────────────────────────────────────
# CLEANUP (Optional but useful)
# ─────────────────────────────────────────────

@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
        del observations[session_id]
        return {"message": "Session deleted"}

    raise HTTPException(status_code=404, detail="Session not found")


# ─────────────────────────────────────────────
# REQUIRED MAIN (for OpenEnv)
# ─────────────────────────────────────────────

def main():
    print("Starting Finance AI Environment...")


if __name__ == "__main__":
    main()
