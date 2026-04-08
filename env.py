
import random
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────

class UpcomingExpense(BaseModel):
    name: str
    amount: float
    is_necessary: bool  # rent/bills vs impulse buys
    scheduled_day: int  # day of month this expense is due


class Observation(BaseModel):
    day_of_month: int                          # 1–30
    balance: float                             # current bank balance
    monthly_income: float                      # fixed monthly salary
    spent_so_far: float                        # cumulative spend this month
    spending_ratio: float                      # spent_so_far / monthly_income
    upcoming_expenses: List[UpcomingExpense]   # expenses in next 7 days
    risk_score: float                          # 0–1; how likely user overspends today
    user_behavior_type: str                    # impulsive | disciplined | emotional
    user_satisfaction: float                   # 0–1; how happy the user is


class Action(BaseModel):
    action_type: str    # allow_spending | block_spending | delay_spending |
                        # send_warning | suggest_alternative
    message: Optional[str] = None  # optional message shown to user


class Reward(BaseModel):
    step_reward: float       # reward for this single step
    cumulative_reward: float # total reward so far this episode
    reason: str              # human-readable explanation


# ─────────────────────────────────────────────
# PERSONA DEFINITIONS
# ─────────────────────────────────────────────

PERSONAS = {
    "impulsive": {
        # High base spend probability; reacts moderately to interventions
        "base_spend_prob": 0.75,
        "unnecessary_ratio": 0.70,   # fraction of purchases that are unnecessary
        "spend_range": (20, 250),
        "warning_sensitivity": 0.2,  # how much a warning reduces spend prob
        "block_satisfaction_hit": 0.08,
        "delay_satisfaction_hit": 0.03,
    },
    "disciplined": {
        "base_spend_prob": 0.30,
        "unnecessary_ratio": 0.20,
        "spend_range": (10, 80),
        "warning_sensitivity": 0.5,
        "block_satisfaction_hit": 0.04,
        "delay_satisfaction_hit": 0.01,
    },
    "emotional": {
        # Spends more on weekends (days 6,7,13,14,20,21,27,28) and mid-month
        "base_spend_prob": 0.50,
        "unnecessary_ratio": 0.55,
        "spend_range": (15, 200),
        "warning_sensitivity": 0.35,
        "block_satisfaction_hit": 0.06,
        "delay_satisfaction_hit": 0.02,
    },
}


# ─────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────

class PersonalFinanceEnv:

    VALID_ACTIONS = {
        "allow_spending",
        "block_spending",
        "delay_spending",
        "send_warning",
        "suggest_alternative",
    }

    def __init__(
        self,
        monthly_income: float = 5000.0,
        behavior_type: str = "impulsive",
        seed: Optional[int] = None,
    ):
        assert behavior_type in PERSONAS, f"Unknown persona: {behavior_type}"
        self.monthly_income = monthly_income
        self.behavior_type = behavior_type
        self.seed = seed
        self._rng = random.Random(seed)

        # Internal tracking (set in reset)
        self._day: int = 0
        self._balance: float = 0.0
        self._spent_so_far: float = 0.0
        self._user_satisfaction: float = 1.0
        self._cumulative_reward: float = 0.0
        self._warning_count: int = 0          # spam-penalty tracker
        self._block_count: int = 0            # excessive-blocking tracker
        self._unnecessary_blocked: int = 0    # successful interventions
        self._upcoming_expenses: List[UpcomingExpense] = []
        self._done: bool = False

    # ── helpers ──────────────────────────────

    def _generate_monthly_expenses(self) -> List[UpcomingExpense]:
        """Pre-generate all potential expenses for the month."""
        persona = PERSONAS[self.behavior_type]
        lo, hi = persona["spend_range"]
        expenses = []

        # Necessary fixed expenses (rent on day 1, bills on day 15)
        expenses.append(UpcomingExpense(
            name="Rent", amount=self._rng.uniform(1000, 1500),
            is_necessary=True, scheduled_day=1
        ))
        expenses.append(UpcomingExpense(
            name="Utilities", amount=self._rng.uniform(80, 150),
            is_necessary=True, scheduled_day=15
        ))
        expenses.append(UpcomingExpense(
            name="Groceries", amount=self._rng.uniform(200, 350),
            is_necessary=True, scheduled_day=7
        ))

        # Discretionary expenses spread across the month
        n_discretionary = self._rng.randint(8, 15)
        for _ in range(n_discretionary):
            day = self._rng.randint(1, 30)
            amount = self._rng.uniform(lo, hi)
            is_necessary = self._rng.random() > persona["unnecessary_ratio"]
            names_unnec = ["Online Shopping", "Restaurant", "Coffee", "Subscription",
                           "Entertainment", "Impulse Buy", "Gadget", "Clothing"]
            names_nec = ["Pharmacy", "Transport", "Internet Bill"]
            name = self._rng.choice(names_nec if is_necessary else names_unnec)
            expenses.append(UpcomingExpense(
                name=name, amount=round(amount, 2),
                is_necessary=is_necessary, scheduled_day=day
            ))

        return expenses

    def _get_upcoming(self) -> List[UpcomingExpense]:
        """Return expenses scheduled within the next 7 days."""
        return [
            e for e in self._upcoming_expenses
            if self._day <= e.scheduled_day <= self._day + 7
        ]

    def _compute_risk_score(self) -> float:
        """
        Estimate probability user will overspend today.
        Emotional persona has elevated risk on 'weekend' days.
        """
        persona = PERSONAS[self.behavior_type]
        base = persona["base_spend_prob"]

        if self.behavior_type == "emotional":
            # Simulate weekend effect every 6-7 days
            if self._day % 7 in (0, 6):
                base = min(1.0, base + 0.25)
            # Mid-month emotional dip
            if 13 <= self._day <= 16:
                base = min(1.0, base + 0.15)

        # Approaching end of month: impulsive types splurge
        if self.behavior_type == "impulsive" and self._day >= 25:
            base = min(1.0, base + 0.15)

        # Clip to [0, 1]
        return round(min(1.0, max(0.0, base)), 4)

    def _build_observation(self) -> Observation:
        spending_ratio = self._spent_so_far / self.monthly_income
        return Observation(
            day_of_month=self._day,
            balance=round(self._balance, 2),
            monthly_income=self.monthly_income,
            spent_so_far=round(self._spent_so_far, 2),
            spending_ratio=round(spending_ratio, 4),
            upcoming_expenses=self._get_upcoming(),
            risk_score=self._compute_risk_score(),
            user_behavior_type=self.behavior_type,
            user_satisfaction=round(self._user_satisfaction, 4),
        )

    # ── OpenEnv API ───────────────────────────

    def reset(self) -> Observation:
        """Reset the environment for a new episode."""
        self._rng = random.Random(self.seed)  # reproducible
        self._day = 1
        self._balance = self.monthly_income   # start with a full month's income
        self._spent_so_far = 0.0
        self._user_satisfaction = 1.0
        self._cumulative_reward = 0.0
        self._warning_count = 0
        self._block_count = 0
        self._unnecessary_blocked = 0
        self._upcoming_expenses = self._generate_monthly_expenses()
        self._done = False
        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Advance the environment by one day.

        Returns:
            observation: new state after action
            reward:      Reward model with step and cumulative reward
            done:        True if month is complete
            info:        debug dictionary
        """
        assert not self._done, "Episode is done. Call reset()."
        assert action.action_type in self.VALID_ACTIONS, \
            f"Invalid action: {action.action_type}"

        persona = PERSONAS[self.behavior_type]
        step_reward = 0.0
        info: Dict[str, Any] = {"day": self._day, "action": action.action_type}

        # ── Today's expenses ──────────────────
        todays_expenses = [
            e for e in self._upcoming_expenses
            if e.scheduled_day == self._day
        ]

        # Determine which expenses actually happen after agent action
        actual_spend = 0.0
        blocked_unnecessary = 0
        allowed_necessary = 0

        for expense in todays_expenses:
            user_wants_to_spend = self._rng.random() < self._compute_risk_score()

            if not user_wants_to_spend and not expense.is_necessary:
                # User didn't even try to spend — no action needed
                continue

            if expense.is_necessary:
                # Necessary expenses always go through (blocking them hurts satisfaction)
                if action.action_type == "block_spending":
                    # Agent blocked a necessary expense — bad!
                    self._user_satisfaction = max(0.0, self._user_satisfaction - 0.10)
                    step_reward -= 2.0
                    info["blocked_necessary"] = True
                actual_spend += expense.amount
                allowed_necessary += 1

            else:
                # Unnecessary expense — agent can intervene
                if action.action_type == "allow_spending":
                    actual_spend += expense.amount

                elif action.action_type == "block_spending":
                    self._block_count += 1
                    blocked_unnecessary += 1
                    # Small satisfaction hit for being blocked
                    self._user_satisfaction = max(
                        0.0,
                        self._user_satisfaction - persona["block_satisfaction_hit"]
                    )
                    step_reward += 1.5  # reward for preventing unnecessary spend

                elif action.action_type == "delay_spending":
                    # Reschedule 3 days ahead (may still happen)
                    expense.scheduled_day = min(30, expense.scheduled_day + 3)
                    self._user_satisfaction = max(
                        0.0,
                        self._user_satisfaction - persona["delay_satisfaction_hit"]
                    )
                    step_reward += 0.5

                elif action.action_type == "send_warning":
                    self._warning_count += 1
                    # Warning reduces spend probability for this expense
                    if self._rng.random() < persona["warning_sensitivity"]:
                        blocked_unnecessary += 1
                        step_reward += 0.8
                    else:
                        actual_spend += expense.amount
                    # Spam penalty if too many warnings
                    if self._warning_count > 5:
                        step_reward -= 0.3
                        self._user_satisfaction = max(
                            0.0, self._user_satisfaction - 0.02
                        )

                elif action.action_type == "suggest_alternative":
                    # Reduces spend by 40% (user accepts cheaper option)
                    reduced = expense.amount * 0.60
                    actual_spend += reduced
                    step_reward += 0.4

        # ── Update financial state ────────────
        self._spent_so_far += actual_spend
        self._balance -= actual_spend
        self._unnecessary_blocked += blocked_unnecessary

        # ── Dense step rewards ────────────────
        spending_ratio = self._spent_so_far / self.monthly_income

        # Reward for staying under 75% budget
        if spending_ratio < 0.75:
            step_reward += 0.3

        # Penalty for exceeding 75% budget
        if spending_ratio > 0.75:
            overshoot = spending_ratio - 0.75
            step_reward -= overshoot * 5.0

        # Penalty if satisfaction drops too low
        if self._user_satisfaction < 0.4:
            step_reward -= 1.0

        # Penalty for excessive blocking (frustrates users)
        if self._block_count > 8:
            step_reward -= 0.5

        # ── End-of-month final reward ─────────
        done = self._day >= 30
        if done:
            self._done = True
            savings = self.monthly_income - self._spent_so_far
            savings_ratio = savings / self.monthly_income

            # Savings bonus (scaled)
            if savings_ratio >= 0.25:
                step_reward += 10.0
            elif savings_ratio >= 0.10:
                step_reward += 5.0
            elif savings_ratio >= 0.0:
                step_reward += 1.0
            else:
                step_reward -= 10.0  # went into debt

            # Satisfaction bonus
            step_reward += self._user_satisfaction * 3.0

        self._cumulative_reward += step_reward
        self._day += 1

        reward = Reward(
            step_reward=round(step_reward, 4),
            cumulative_reward=round(self._cumulative_reward, 4),
            reason=f"Day {self._day - 1}: spent={actual_spend:.2f}, "
                   f"ratio={spending_ratio:.2%}, sat={self._user_satisfaction:.2f}",
        )

        info.update({
            "actual_spend": actual_spend,
            "spending_ratio": spending_ratio,
            "user_satisfaction": self._user_satisfaction,
            "blocked_unnecessary": blocked_unnecessary,
            "cumulative_reward": self._cumulative_reward,
        })

        return self._build_observation(), reward, done, info

    def state(self) -> Dict[str, Any]:
        """Return the full internal state (for debugging/logging)."""
        return {
            "day_of_month": self._day,
            "balance": self._balance,
            "monthly_income": self.monthly_income,
            "spent_so_far": self._spent_so_far,
            "spending_ratio": self._spent_so_far / self.monthly_income,
            "user_satisfaction": self._user_satisfaction,
            "behavior_type": self.behavior_type,
            "warning_count": self._warning_count,
            "block_count": self._block_count,
            "unnecessary_blocked": self._unnecessary_blocked,
            "cumulative_reward": self._cumulative_reward,
            "done": self._done,
        }
    