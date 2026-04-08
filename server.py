from fastapi import FastAPI
from env import PersonalFinanceEnv
from inference import RuleBasedAgent

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Finance AI Env is running"}

@app.post("/run")
def run_agent():
    env = PersonalFinanceEnv()
    agent = RuleBasedAgent()

    obs = env.reset()
    done = False

    while not done:
        action = agent(obs)
        obs, reward, done, _ = env.step(action)

    return {
        "final_balance": obs.balance,
        "reward": reward
    }