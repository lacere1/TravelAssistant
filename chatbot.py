from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Example:
    pattern: str
    response: str


class TrafficChatbot:
    def __init__(self) -> None:
        self.examples: List[Example] = [
            Example(
                "bus",
                "Tell me which stop or area you care about and I’ll help you think about bus times and routes.",
            ),
            Example(
                "train",
                "Ask about a station or line and I’ll help you reason about train times and status.",
            ),
            Example(
                "status",
                "Mention a bus route or train line and I’ll focus on its status and disruptions.",
            ),
            Example(
                "hello",
                "Hi! Ask me about bus schedules, train schedules, bus status, train status, or journey planning.",
            ),
        ]

    def reply(self, message: str) -> str:
        text = (message or "").lower()
        if not text:
            return "Say something like: “Bus times for Oxford Circus”, “Victoria line status”, or “Plan a journey from Harlesden to Oxford Street.”"

        for ex in self.examples:
            if ex.pattern in text:
                return ex.response

        return (
            "I’m your travel assistant. I can help you with queries based on the London network, including bus schedules, train schedules, "
            "bus status, train status, and journey planning around London."
        )

