# 💰 Personal Finance Control — OpenEnv RL Environment

> **Hackathon Submission** | OpenEnv-compliant | Python 3.11 | Pydantic v2

---
---
title: Finance AI Environment
emoji: 💰
colorFrom: blue
colorTo: green
sdk: docker
app_file: inference.py
pinned: false
---
## Problem Statement

Personal finance is one of the hardest behavioural challenges humans face. Most people _know_ they should save more, yet impulse purchases, emotional spending, and cognitive biases consistently derail even the best intentions.

**The gap:** Existing personal finance apps provide dashboards and alerts, but they are _reactive_ — they tell you what happened, not what to do next. They also treat all users identically, ignoring that an impulsive spender needs very different nudges than a disciplined one.

---

## Why Current Solutions Fail

| Problem | Current Apps | This Environment |
|---|---|---|
| Generic advice | One-size-fits-all alerts | 3 distinct user personas |
| Reactive only | Shows past spending | Agent acts _before_ the purchase |
| No learning | Static rule engine | RL agent improves over episodes |
| No trade-offs | Just "spend less" | Balances savings _and_ satisfaction |
| No evaluation | No ground truth | Deterministic grader, score [0, 1] |

---

## Environment Overview

The environment simulates **one calendar month (30 days)** of a user's financial life. Each day, the agent:

1. Receives an **Observation** (balance, spending ratio, upcoming bills, risk score, satisfaction).
2. Chooses an **Action** (allow, block, delay, warn, suggest alternative).
3. Receives a **dense Reward** based on the outcome.

The user is one of three **personas** that react differently to interventions, making the problem non-trivial.

```
Day 1 ──► Agent observes state ──► Agent acts ──► Environment updates
Day 2 ──► ...
...
Day 30 ──► Terminal reward (savings + satisfaction bonus)
```

---

## Observation Space

| Field | Type | Range | Description |
|---|---|---|---|
| `day_of_month` | int | 1–30 | Current simulation day |
| `balance` | float | ≥ 0 | Bank balance in USD |
| `monthly_income` | float | fixed | Salary for the month |
| `spent_so_far` | float | ≥ 0 | Cumulative spend |
| `spending_ratio` | float | 0–2+ | `spent / income` |
| `upcoming_expenses` | list | — | Bills due in next 7 days |
| `risk_score` | float | 0–1 | Overspend probability today |
| `user_behavior_type` | str | enum | Active persona |
| `user_satisfaction` | float | 0–1 | User happiness |

---

## Action Space

| Action | Effect | Satisfaction Impact |
|---|---|---|
| `allow_spending` | Purchase proceeds normally | None |
| `block_spending` | Purchase is prevented | −0.04 to −0.08 per block |
| `delay_spending` | Rescheduled +3 days | −0.01 to −0.03 |
| `send_warning` | Educational nudge sent | −0.02 if spammed (>5 warnings) |
| `suggest_alternative` | 40% cost reduction | Minimal |

Each action can carry an optional `message` string shown to the user.

---

## Reward Design

The reward is **dense** — feedback is given every step, not just at the end.

### Step Rewards

| Event | Reward |
|---|---|
| spending_ratio < 0.75 | **+0.30** (per day under budget) |
| spending_ratio > 0.75 | **−(overshoot × 5.0)** |
| Unnecessary purchase blocked | **+1.50** |
| Warning successfully deters spending | **+0.80** |
| Expense delayed | **+0.50** |
| Cheaper alternative accepted | **+0.40** |
| Necessary expense blocked | **−2.00** (agent error) |
| Warning spam (>5 warnings) | **−0.30** per extra warning |
| Excessive blocking (>8 blocks) | **−0.50** |
| Satisfaction drops below 0.4 | **−1.00** per day |

### Terminal Rewards (Day 30)

| Savings Achieved | Bonus |
|---|---|
| ≥ 25% of income | **+10.0** |
| ≥ 10% of income | **+5.0** |
| ≥ 0% (no debt) | **+1.0** |
| Went into debt | **−10.0** |
| Satisfaction bonus | **+satisfaction × 3.0** |

**Design philosophy:** The agent cannot simply block everything — satisfaction would collapse and trigger its own penalty. It must learn _when_ to intervene and with _which_ action.

---

## User Personas

### 🔴 Impulsive Spender
- Base spend probability: **75%**
- 70% of purchases are unnecessary
- Reacts weakly to warnings (20% effectiveness)
- High satisfaction hit when blocked

### 🟢 Disciplined User
- Base spend probability: **30%**
- Only 20% unnecessary purchases
- Responds well to warnings (50% effectiveness)
- Tolerates light interventions

### 🟡 Emotional Spender
- Base spend probability: **50%** (elevated on "weekends" and mid-month)
- 55% unnecessary purchases
- Moderate warning effectiveness (35%)
- Moderate satisfaction sensitivity

---

## Task Descriptions

### Task 1 — Budget Compliance (Easy)
- **Persona:** Disciplined | **Seed:** 42 | **Income:** $5,000
- **Goal:** Keep `spending_ratio` below 0.75
- **Grader:** `score = 1.0` if ratio ≤ 0.75, linearly decreasing to 0.0 at 100% spend

### Task 2 — Impulse Control (Medium)
- **Persona:** Impulsive | **Seed:** 99 | **Income:** $4,000
- **Goal:** Intervene on spending spikes, save as much as possible
- **Grader:** `score = 0.6 × savings_ratio + 0.4 × satisfaction`

### Task 3 — Full Month Optimisation (Hard)
- **Persona:** Emotional | **Seed:** 777 | **Income:** $6,000
- **Goal:** Maximise _both_ savings and satisfaction over 30 days
- **Grader:** `score = 0.5 × savings_ratio + 0.5 × satisfaction` (halved if satisfaction < 0.50)

---

## Evaluation Method

All tasks use **fixed seeds** → scores are fully reproducible.

```
final_score = mean(task1_score, task2_score, task3_score)
```

Scores are in **[0.0, 1.0]**. Higher is better.

---

## Setup and Run

### Local (Python)

```bash
# 1. Clone / unzip the project
cd personal-finance-control

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run baseline evaluation
python inference.py
```

### Docker

```bash
# Build
docker build -t finenv .

# Run
docker run --rm finenv
```

---

## Example Output

```
============================================================
  Personal Finance RL Environment — Baseline Evaluation
============================================================

▶ Evaluating: RandomAgent (seed=0)
------------------------------------------------------------
  [Task 1 — Budget Compliance (Easy)]    Score: 0.4123 | Spent: 4521.30 / 5000.00 | Satisfaction: 0.72
  [Task 2 — Impulse Control (Medium)]    Score: 0.3801 | Spent: 3812.45 / 4000.00 | Satisfaction: 0.61
  [Task 3 — Full Month Optimisation]     Score: 0.2950 | Spent: 5944.20 / 6000.00 | Satisfaction: 0.48

▶ Evaluating: RuleBasedAgent
------------------------------------------------------------
  [Task 1 — Budget Compliance (Easy)]    Score: 0.9800 | Spent: 3650.00 / 5000.00 | Satisfaction: 0.88
  [Task 2 — Impulse Control (Medium)]    Score: 0.7120 | Spent: 2100.80 / 4000.00 | Satisfaction: 0.74
  [Task 3 — Full Month Optimisation]     Score: 0.6430 | Spent: 3900.10 / 6000.00 | Satisfaction: 0.65

============================================================
  SUMMARY
============================================================
Agent                        T1      T2      T3     Avg
------------------------------------------------------------
RandomAgent (seed=0)     0.4123  0.3801  0.2950  0.3625
RuleBasedAgent           0.9800  0.7120  0.6430  0.7783
============================================================
```

---

## Project Structure

```
personal-finance-control/
├── env.py           # Core RL environment (OpenEnv API)
├── tasks.py         # 3 evaluation tasks + graders
├── inference.py     # Baseline agents + evaluation runner
├── openenv.yaml     # OpenEnv configuration schema
├── Dockerfile       # Container definition
├── requirements.txt # Python dependencies
└── README.md        # This file
```

---

## Extending This Environment

- **Add personas:** Extend the `PERSONAS` dict in `env.py`
- **Add actions:** Add to `VALID_ACTIONS` and handle in `step()`
- **Custom reward:** Modify the reward sections in `step()`
- **New tasks:** Subclass the task pattern in `tasks.py`
- **RL training:** Wrap `PersonalFinanceEnv` in a Gym-compatible adapter

---

## License

MIT License — free to use, modify, and distribute.