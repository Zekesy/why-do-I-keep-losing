from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PlayerPick:
    hero_id: int
    team: str  # "radiant" or "dire"
    role: Optional[int]  # 1: Core/Safe, 2: Mid, 3: Off, 4: Soft Sup, 5: Hard Sup
    is_winner: bool


@dataclass
class Ban:
    hero_id: int
    team: str  # "radiant" or "dire"
    order: int


@dataclass
class MatchSummary
    import torch:
    match_id: int
    patch: Optional[int]
    winning_team: str  # "radiant" or "dire"
    picks: List[PlayerPick] = field(default_factory=list)
    bans: List[Ban] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MatchSummary":
        match_id = data["match_id"]
        radiant_win = data.get("radiant_win", False)
        winning_team = "radiant" if radiant_win else "dire"

        # 1. Parse Player Picks & Roles
        picks = []
        for player in data.get("players", []):
            # player_slot < 128 is Radiant, >= 128 is Dire
            player_team = "radiant" if player.get("player_slot", 0) < 128 else "dire"
            p_win = player_team == winning_team

            picks.append(
                PlayerPick(
                    hero_id=player.get("hero_id", 0),
                    team=player_team,
                    role=player.get("lane_role"),  # OpenDota lane role estimation
                    is_winner=p_win,
                )
            )

        # 2. Parse Bans from draft sequence
        bans = []
        for pb in data.get("picks_bans") or []:
            if not pb.get("is_pick", False):  # Filter for bans only
                ban_team = "radiant" if pb.get("team") == 0 else "dire"
                bans.append(
                    Ban(
                        hero_id=pb.get("hero_id", 0),
                        team=ban_team,
                        order=pb.get("order", 0),
                    )
                )

        return cls(
            match_id=match_id,
            patch=data.get("patch"),
            winning_team=winning_team,
            picks=picks,
            bans=bans,
        )
