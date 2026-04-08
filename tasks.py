
from typing import Callable, Dict, Any, Tuple
from env import PersonalFinanceEnv, Action, Observation


# ─────────────────────────────────────────────
# TYPE ALIAS
# ─────────────────────────────────────────────

AgentFn = Callable[[Observation], Action]
EpisodeResult = Dict[str, Any]


# ─────────────────────────────────────────────
# SHARED RUNNER
# ─────────────────────────────────────────────

def run_episode(env: PersonalFinanceEnv, agent: AgentFn) -> EpisodeResult:
    obs = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        action = agent(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward.step_reward
        steps += 1

    final = env.state()
    final["total_reward"] = total_reward
    final["steps"] = steps
    return final


# ─────────────────────────────────────────────
# TASK 1 — EASY
# ─────────────────────────────────────────────

class Task1:
    
    NAME = "Task 1 — Budget Compliance (Easy)"
    SEED = 42
    INCOME = 5000.0
    THRESHOLD = 0.75   # 75% of income

    @classmethod
    def make_env(cls) -> PersonalFinanceEnv:
        return PersonalFinanceEnv(
            monthly_income=cls.INCOME,
            behavior_type="disciplined",
            seed=cls.SEED,
        )

    @staticmethod
    def run(agent: AgentFn) -> EpisodeResult:
        env = Task1.make_env()
        return run_episode(env, agent)

    @staticmethod
    def grader(result: EpisodeResult) -> float:
        
        ratio = result["spending_ratio"]
        if ratio <= Task1.THRESHOLD:
            return 1.0
        elif ratio >= 1.0:
            return 0.0
        else:
            # Linear penalty between 75% and 100%
            return round(1.0 - (ratio - Task1.THRESHOLD) / (1.0 - Task1.THRESHOLD), 4)


# ─────────────────────────────────────────────
# TASK 2 — MEDIUM
# ─────────────────────────────────────────────

class Task2:


    NAME = "Task 2 — Impulse Control (Medium)"
    SEED = 99
    INCOME = 4000.0

    @classmethod
    def make_env(cls) -> PersonalFinanceEnv:
        return PersonalFinanceEnv(
            monthly_income=cls.INCOME,
            behavior_type="impulsive",
            seed=cls.SEED,
        )

    @staticmethod
    def run(agent: AgentFn) -> EpisodeResult:
        env = Task2.make_env()
        return run_episode(env, agent)

    @staticmethod
    def grader(result: EpisodeResult) -> float:

        income = Task2.INCOME
        savings = income - result["spent_so_far"]
        savings_ratio = max(0.0, min(1.0, savings / income))

        satisfaction = max(0.0, min(1.0, result["user_satisfaction"]))

        score = 0.60 * savings_ratio + 0.40 * satisfaction
        return round(score, 4)


# ─────────────────────────────────────────────
# TASK 3 — HARD
# ─────────────────────────────────────────────

class Task3:
    

    NAME = "Task 3 — Full Month Optimisation (Hard)"
    SEED = 777
    INCOME = 6000.0
    MIN_SATISFACTION = 0.50  # if below this, score is penalised

    @classmethod
    def make_env(cls) -> PersonalFinanceEnv:
        return PersonalFinanceEnv(
            monthly_income=cls.INCOME,
            behavior_type="emotional",
            seed=cls.SEED,
        )

    @staticmethod
    def run(agent: AgentFn) -> EpisodeResult:
        env = Task3.make_env()
        return run_episode(env, agent)

    @staticmethod
    def grader(result: EpisodeResult) -> float:

        income = Task3.INCOME
        savings = income - result["spent_so_far"]
        savings_score = max(0.0, min(1.0, savings / income))
        sat_score = max(0.0, min(1.0, result["user_satisfaction"]))

        base_score = 0.50 * savings_score + 0.50 * sat_score

        # Hard penalty if satisfaction dropped below minimum
        if sat_score < Task3.MIN_SATISFACTION:
            base_score *= 0.5

        return round(base_score, 4)


# ─────────────────────────────────────────────
# CONVENIENCE: RUN ALL TASKS
# ─────────────────────────────────────────────

ALL_TASKS = [Task1, Task2, Task3]


def evaluate_agent(agent: AgentFn) -> Dict[str, float]:
    
    scores = {}
    for task in ALL_TASKS:
        result = task.run(agent)
        score = task.grader(result)
        scores[task.NAME] = score
        print(f"  [{task.NAME}] Score: {score:.4f} | "
              f"Spent: {result['spent_so_far']:.2f} / {result['monthly_income']:.2f} | "
              f"Satisfaction: {result['user_satisfaction']:.2f}")

    avg = sum(scores.values()) / len(scores)
    scores["average"] = round(avg, 4)
    return scores