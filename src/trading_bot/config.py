from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
import os


@dataclass
class Config:
    alpaca_api_key: str
    alpaca_secret_key: str
    anthropic_api_key: str
    trading_mode: str  # "paper" or "live"
    auto_execute: bool
    claude_model: str
    max_position_size: float
    max_daily_loss: float
    min_confidence: float

    @property
    def is_paper(self) -> bool:
        return self.trading_mode == "paper"

    @property
    def alpaca_base_url(self) -> str:
        if self.is_paper:
            return "https://paper-api.alpaca.markets"
        return "https://api.alpaca.markets"


def load_config() -> Config:
    load_dotenv(Path.cwd() / ".env")

    return Config(
        alpaca_api_key=os.environ["ALPACA_API_KEY"],
        alpaca_secret_key=os.environ["ALPACA_SECRET_KEY"],
        anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
        trading_mode=os.getenv("TRADING_MODE", "paper"),
        auto_execute=os.getenv("AUTO_EXECUTE", "false").lower() == "true",
        claude_model=os.getenv("CLAUDE_MODEL", "claude-haiku-4-5"),
        max_position_size=float(os.getenv("MAX_POSITION_SIZE", "1000")),
        max_daily_loss=float(os.getenv("MAX_DAILY_LOSS", "500")),
        min_confidence=float(os.getenv("MIN_CONFIDENCE", "0.30")),
    )
