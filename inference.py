
import random
from env import Action, Observation
from tasks import evaluate_agent, ALL_TASKS


# ─────────────────────────────────────────────
# AGENT 1: RANDOM
# ─────────────────────────────────────────────

class RandomAgent:
    """Picks a random action every step. Useful as a lower-bound baseline."""

    ACTIONS = [
        "allow_spending",
        "block_spending",
        "delay_spending",
        "send_warning",
        "suggest_alternative",
    ]

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def __call__(self, obs: Observation) -> Action:
        return Action(
            action_type=self._rng.choice(self.ACTIONS),
            message="Random agent action",
        )


# ─────────────────────────────────────────────
# AGENT 2: RULE-BASED
# ─────────────────────────────────────────────

class RuleBasedAgent:
    """
    Heuristic agent with clear decision rules:

      spending_ratio > 0.80  → block_spending  (critical overspend)
      spending_ratio > 0.70  → send_warning    (approaching limit)
      risk_score     > 0.70  → delay_spending  (high-risk day)
      risk_score     > 0.50  → suggest_alternative (offer cheaper option)
      otherwise              → allow_spending
    """

    def __call__(self, obs: Observation) -> Action:
        ratio = obs.spending_ratio
        risk = obs.risk_score

        if ratio > 0.80:
            return Action(
                action_type="block_spending",
                message="You've exceeded 80% of your monthly budget. "
                        "This purchase has been blocked.",
            )
        elif ratio > 0.70:
            return Action(
                action_type="send_warning",
                message=f"You've used {ratio:.0%} of your monthly budget. "
                        "Consider slowing down.",
            )
        elif risk > 0.70:
            return Action(
                action_type="delay_spending",
                message="High-risk spending day detected. "
                        "Let's schedule this for later.",
            )
        elif risk > 0.50:
            return Action(
                action_type="suggest_alternative",
                message="There might be a cheaper way to meet this need.",
            )
        else:
            return Action(
                action_type="allow_spending",
                message="Spending looks fine today.",
            )
class AdaptiveAgent:
    """
    Smarter agent that balances savings and satisfaction dynamically.
    """

    def __call__(self, obs: Observation) -> Action:
        ratio = obs.spending_ratio
        risk = obs.risk_score
        satisfaction = obs.user_satisfaction

        # If user is unhappy → relax control
        if satisfaction < 0.5:
            return Action(
                action_type="allow_spending",
                message="Relaxing restrictions to maintain satisfaction.",
            )

        # If overspending → strict control
        if ratio > 0.75:
            return Action(
                action_type="block_spending",
                message="Spending exceeded safe limit. Blocking.",
            )

        # High risk → delay
        if risk > 0.65:
            return Action(
                action_type="delay_spending",
                message="High-risk spending detected. Delaying.",
            )

        # Medium risk → cheaper option
        if risk > 0.45:
            return Action(
                action_type="suggest_alternative",
                message="Suggesting a cheaper alternative.",
            )

        return Action(
            action_type="allow_spending",
            message="Spending is safe.",
        )


# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Personal Finance RL Environment — Baseline Evaluation")
    print("=" * 60)

    agents = {
    "RandomAgent (seed=0)": RandomAgent(seed=0),
    "RuleBasedAgent":       RuleBasedAgent(),
    "AdaptiveAgent":        AdaptiveAgent(),
}

    all_results = {}

    for agent_name, agent in agents.items():
        print(f"\n▶ Evaluating: {agent_name}")
        print("-" * 60)
        scores = evaluate_agent(agent)
        all_results[agent_name] = scores

    # ── Summary table ─────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    task_names = [t.NAME for t in ALL_TASKS]
    header = f"{'Agent':<25}" + "".join(f"{'T' + str(i+1):>8}" for i in range(3)) + f"{'Avg':>8}"
    print(header)
    print("-" * 60)

    for agent_name, scores in all_results.items():
        row = f"{agent_name:<25}"
        for tname in task_names:
            row += f"{scores[tname]:>8.4f}"
        row += f"{scores['average']:>8.4f}"
        print(row)

    print("=" * 60)
    print("\nNotes:")
    print("  T1 = Budget Compliance (Easy)    — disciplined user")
    print("  T2 = Impulse Control   (Medium)  — impulsive user")
    print("  T3 = Full Optimisation (Hard)    — emotional user")
    print("  Scores are in [0.0, 1.0]. Higher is better.")
    print("  All runs are deterministic (fixed seeds).")
    print("\n" + "=" * 60)
    print("  INSIGHTS")
    print("=" * 60)
    print("- RandomAgent performs surprisingly well due to low intervention penalties.")
    print("- RuleBasedAgent tends to over-control, reducing user satisfaction.")
    print("- AdaptiveAgent balances savings and satisfaction more effectively.")
    print("- This demonstrates the importance of behavior-aware financial decision systems.")


if __name__ == "__main__":
    main()