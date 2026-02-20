from trading_bot.config import Config
from trading_bot.models import Action, SafetyCheckResult, TradeDecision


def check_confidence(config: Config, decision: TradeDecision) -> SafetyCheckResult:
    if decision.action == Action.HOLD:
        return SafetyCheckResult(passed=True)
    if decision.confidence_score < config.min_confidence:
        return SafetyCheckResult(
            passed=False,
            reason=f"Confidence {decision.confidence_score:.0%} below minimum {config.min_confidence:.0%}",
        )
    return SafetyCheckResult(passed=True)


def check_position_size(config: Config, decision: TradeDecision) -> SafetyCheckResult:
    if decision.action == Action.HOLD:
        return SafetyCheckResult(passed=True)
    position_value = decision.suggested_shares * decision.suggested_entry_price
    if position_value > config.max_position_size:
        return SafetyCheckResult(
            passed=False,
            reason=f"Position size ${position_value:.2f} exceeds max ${config.max_position_size:.2f}",
        )
    return SafetyCheckResult(passed=True)


def check_risk_reward(decision: TradeDecision) -> SafetyCheckResult:
    if decision.action == Action.HOLD:
        return SafetyCheckResult(passed=True)
    if decision.risk_reward_ratio < 1.5:
        return SafetyCheckResult(
            passed=False,
            reason=f"Risk/reward {decision.risk_reward_ratio:.2f}:1 below minimum 1.5:1",
        )
    return SafetyCheckResult(passed=True)


def run_safety_checks(config: Config, decision: TradeDecision) -> list[SafetyCheckResult]:
    checks = [
        check_confidence(config, decision),
        check_position_size(config, decision),
        check_risk_reward(decision),
    ]
    return checks


def all_checks_pass(checks: list[SafetyCheckResult]) -> bool:
    return all(c.passed for c in checks)
