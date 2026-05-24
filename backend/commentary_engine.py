"""
Commentary Engine
=================
Generates real-time fight commentary from classified moves.
Uses template pools with contextual awareness for combos and exchanges.
"""

import random
from typing import List, Dict, Optional
from dataclasses import dataclass
from .move_classifier import MoveClassification
from .utils import setup_logging

logger = setup_logging(__name__)


@dataclass
class CommentaryEvent:
    """A single line of generated commentary."""

    text: str
    timestamp: float
    event_type: str          # "opening" | "move" | "combo" | "exchange"
    players_involved: List[int]


class CommentaryEngine:
    """
    Generates contextual fight commentary from a stream of move events.
    Tracks recent activity to detect combos and exchanges, and enforces
    a minimum gap between consecutive commentary lines.
    """

    MOVE_LINES = {
        "jab": [
            "Player {p} stings with a quick jab!",
            "Sharp jab from Player {p}.",
            "Player {p} probes the range with a jab.",
            "Clean jab from Player {p}!",
        ],
        "cross": [
            "Player {p} unloads a heavy cross!",
            "Big straight right from Player {p}!",
            "Player {p} connects with a powerful cross!",
            "Boom — hard cross by Player {p}!",
        ],
        "uppercut": [
            "Player {p} digs in with an uppercut!",
            "Devastating uppercut from Player {p}!",
            "Player {p} lands a picture-perfect uppercut!",
            "That uppercut from Player {p} had serious pop!",
        ],
        "front_kick": [
            "Player {p} fires a front kick to create space!",
            "Nice teep from Player {p}!",
            "Player {p} uses the front kick to control distance.",
            "Front kick to the midsection from Player {p}!",
        ],
        "roundhouse_kick": [
            "Player {p} swings a roundhouse kick!",
            "Powerful Thai-style roundhouse from Player {p}!",
            "Player {p} goes high with the roundhouse!",
            "That roundhouse from Player {p} had serious power behind it!",
        ],
        "neutral": [
            "Both fighters circling, reading each other.",
            "A tactical reset — both fighters reassess.",
        ],
    }

    COMBO_LINES = [
        "Player {p} is putting together a slick combination!",
        "Multiple strikes in quick succession from Player {p}!",
        "Player {p} is letting the hands go!",
        "Great combination work from Player {p}!",
    ]

    EXCHANGE_LINES = [
        "Both fighters are trading! What an exchange!",
        "They're going blow-for-blow — this is heating up!",
        "Neither fighter is backing down here!",
        "What a flurry! Both players connecting!",
    ]

    OPENING_LINES = [
        "Here we go! Both fighters looking sharp from the jump.",
        "The fight is on! Let's see who takes control early.",
        "We're live! Both fighters feeling out the range.",
    ]

    def __init__(
        self,
        min_time_between_comments: float = 2.0,
        combo_window: float = 3.0,
        exchange_window: float = 2.0,
    ):
        self.min_gap = min_time_between_comments
        self.combo_window = combo_window
        self.exchange_window = exchange_window

        self.last_comment_time = 0.0
        self.recent_moves: List[MoveClassification] = []
        self.fight_started = False
        self.move_counts: Dict[int, Dict[str, int]] = {1: {}, 2: {}}
        self.history: List[CommentaryEvent] = []

        logger.info("CommentaryEngine ready")

    # ------------------------------------------------------------------ public

    def generate_commentary(
        self, move: MoveClassification
    ) -> Optional[CommentaryEvent]:
        """
        Decide whether to generate commentary for this move and return it.
        Returns None if the minimum gap hasn't elapsed or the move is neutral.
        """
        if not self.fight_started:
            self.fight_started = True
            line = random.choice(self.OPENING_LINES)
            return self._emit(line, move.timestamp, "opening", [1, 2])

        if move.move_name == "neutral":
            return None

        if move.timestamp - self.last_comment_time < self.min_gap:
            return None

        self.recent_moves.append(move)
        self._update_counts(move)
        self._trim_recent(move.timestamp)

        if self._is_combo(move):
            line = random.choice(self.COMBO_LINES).format(p=move.player_id)
            event = self._emit(line, move.timestamp, "combo", [move.player_id])
        elif self._is_exchange():
            line = random.choice(self.EXCHANGE_LINES)
            event = self._emit(line, move.timestamp, "exchange", [1, 2])
        else:
            pool = self.MOVE_LINES.get(move.move_name, [])
            if not pool:
                return None
            line = random.choice(pool).format(p=move.player_id)
            event = self._emit(line, move.timestamp, "move", [move.player_id])

        self.last_comment_time = move.timestamp
        self.history.append(event)
        return event

    def get_fight_summary(self) -> Dict:
        return {
            "total_commentary_lines": len(self.history),
            "player_1_moves": self.move_counts[1],
            "player_2_moves": self.move_counts[2],
            "total_moves": {
                1: sum(self.move_counts[1].values()),
                2: sum(self.move_counts[2].values()),
            },
        }

    def reset(self):
        self.last_comment_time = 0.0
        self.recent_moves.clear()
        self.fight_started = False
        self.move_counts = {1: {}, 2: {}}
        self.history.clear()
        logger.info("CommentaryEngine reset")

    # ----------------------------------------------------------------- helpers

    def _is_combo(self, move: MoveClassification) -> bool:
        same_player = [
            m for m in self.recent_moves
            if m.player_id == move.player_id
            and move.timestamp - m.timestamp <= self.combo_window
            and m.move_name != "neutral"
        ]
        return len(same_player) >= 2

    def _is_exchange(self) -> bool:
        if len(self.recent_moves) < 2:
            return False
        latest = self.recent_moves[-1].timestamp
        window = [
            m for m in self.recent_moves
            if latest - m.timestamp <= self.exchange_window
            and m.move_name != "neutral"
        ]
        return len({m.player_id for m in window}) == 2 and len(window) >= 3

    def _trim_recent(self, current_time: float):
        cutoff = current_time - max(self.combo_window, self.exchange_window)
        self.recent_moves = [m for m in self.recent_moves if m.timestamp > cutoff]

    def _update_counts(self, move: MoveClassification):
        pid, name = move.player_id, move.move_name
        self.move_counts[pid][name] = self.move_counts[pid].get(name, 0) + 1

    def _emit(
        self, text: str, timestamp: float, event_type: str, players: List[int]
    ) -> CommentaryEvent:
        return CommentaryEvent(
            text=text,
            timestamp=timestamp,
            event_type=event_type,
            players_involved=players,
        )
