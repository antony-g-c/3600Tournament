"""
PlayerAgent — Expectiminimax-based bot with HMM rat tracking.

High-level flow per turn:
  1. Update the rat belief (HMM predict + noise/distance update + opponent/
     self search side-channel).
  2. Decide time budget.
  3. Decide whether we are "desperate" (late + behind) to unlock riskier
     search moves.
  4. Iterative-deepening alpha-beta (PVS + LMR).
  5. Return chosen move.
"""
import time
from collections.abc import Callable
from typing import Tuple

import numpy as np

from game.board import Board

from . import rat_belief
from .expectiminimax import ExpectiMinimax


def _budget_for_turn(board: Board, time_left_fn: Callable) -> float:
    """
    Time to spend on this move.

    Rules:
      * Baseline: remaining / turns_left, with a small over-spend factor so
        early turns (cheap to evaluate) free up time for middlegame.
      * Cap: never more than 12 s on any single turn (avoid blowing the
        time bank on a hard-but-uncertain decision).
      * Floor: 0.15 s so we always get to depth 2-3.
      * Endgame: if < 5 turns left, we care about precision, so bump the
        over-spend factor.
    """
    try:
        remaining = float(time_left_fn())
    except Exception:
        remaining = float(getattr(board.player_worker, "time_left", 240.0))

    turns_left = max(1, int(board.player_worker.turns_left))
    base = remaining / turns_left

    # Over-spend more in middlegame, less in opening (heuristic is flat
    # with no primed cells so deeper search rarely changes the move) and
    # in the closing turns once the game is largely decided.
    turn_num = 40 - turns_left  # 0 at start, 39 at end
    if turn_num < 2:
        mult = 0.6   # first 2 turns: save budget for chain races
    elif turn_num < 4:
        mult = 0.9   # opening window
    elif turns_left <= 4:
        mult = 1.7   # precision matters in the endgame (was 1.6)
    else:
        mult = 1.4   # default middlegame (was 1.25 — tournament logs
                     # showed 85-120s of 240s budget unused per game)

    budget = base * mult
    budget = min(budget, remaining * 0.45, 12.0)
    return max(budget, 0.15)


def _is_desperate(board: Board) -> bool:
    """Behind + late? Unlocks marginal +EV searches in prune_and_order_moves."""
    diff = board.player_worker.points - board.opponent_worker.points
    turns = min(board.player_worker.turns_left, board.opponent_worker.turns_left)
    return (diff <= -3 and turns <= 6) or (diff <= -8 and turns <= 12)


def _is_cautious(board: Board) -> bool:
    """Ahead + late? Raises the root search-EV threshold so we don't burn
    tempo on speculative searches when a safe carpet/plain move preserves
    the lead. Only fires when we're comfortably ahead to avoid losing
    real information when the game is still close."""
    diff = board.player_worker.points - board.opponent_worker.points
    turns = min(board.player_worker.turns_left, board.opponent_worker.turns_left)
    return diff >= 5 and turns <= 5


class PlayerAgent:
    """Expectiminimax bot with HMM rat tracking."""

    def __init__(self, board, transition_matrix, time_left,
                 disable_search: bool = False, cfg=None,
                 max_depth: int = 14, **kwargs):
        from .expectiminimax import ExpectiMinimax, HeuristicConfig
        self.rat_tracker = rat_belief.RatBelief(transition_matrix)
        self.rat_t = np.array(transition_matrix, dtype=np.float64)
        self.disable_search = disable_search

        # max_depth=12 for tournament. This is a CEILING, not a floor --
        # iterative deepening stops when the time budget runs out and only
        # commits to fully-completed search iterations, so raising the
        # ceiling cannot cause regressions. Per-turn probe on seed 0 at
        # max_depth=10 showed we were depth-bound on 38/40 turns and
        # ended the game with 135s of the 240s budget unused. At
        # max_depth=12 the middlegame is time-bound at ~7.5s per turn
        # (completing depth 10-11, same as before) and the endgame
        # reaches depth 12 comfortably. 4/4 smoke-test wins held at
        # depth 12 with +18 avg margin vs depth 10's +14 on the same
        # seeds. Wall time ~175s per game, safely under the 240s budget.
        self.engine = ExpectiMinimax(
            transition_matrix,
            max_depth=max_depth,
            cfg=cfg,
            disable_search=disable_search,
        )

        # Search-miss guardrail. Our belief's p_max is empirically over-
        # confident — iteration-5 tournament showed 0/19 hit rate across
        # 4 losses, bleeding ~9.5 pts/game. After ANY miss (streak >= 1,
        # tightened from 2), suppress searches for the next few turns so
        # the belief can re-concentrate on new sensor data before we risk
        # another -2 penalty.
        self._miss_streak = 0
        self._suppress_until_turn = -1
        self._MISS_SUPPRESS_TURNS = 6     # was 4

    def commentate(self):
        return "Thinking..."

    def play(self, board: Board, sensor_data: Tuple, time_left: Callable):
        """
        sensor_data = (noise_enum, estimated_manhattan_distance)
        """
        # ---- 1. HMM update --------------------------------------------
        noise, estimated_dist = sensor_data
        worker_pos = board.player_worker.get_location()

        if board.turn_count > 0:
            self.rat_tracker.predict()

            op_search_loc, op_search_result = board.opponent_search
            if op_search_loc is not None:
                if op_search_result is True:
                    self.rat_tracker = rat_belief.RatBelief(self.rat_t)
                else:
                    self.rat_tracker.zero_out(op_search_loc)

            my_search_loc, my_search_result = board.player_search
            if my_search_loc is not None:
                if my_search_result is True:
                    self.rat_tracker = rat_belief.RatBelief(self.rat_t)
                    self._miss_streak = 0
                else:
                    self.rat_tracker.zero_out(my_search_loc)
                    self._miss_streak += 1
                    # Tightened from >=2 to >=1: iteration-5 tournament
                    # showed the very first miss was already predictive
                    # of continued belief miscalibration.
                    if self._miss_streak >= 1:
                        self._suppress_until_turn = (
                            board.turn_count + self._MISS_SUPPRESS_TURNS
                        )

        floor_types = self.rat_tracker.get_floor_types_array(board)
        self.rat_tracker.update_vectorized(
            floor_types, noise, estimated_dist, worker_pos
        )

        # ---- 2. Decide search parameters ------------------------------
        budget = _budget_for_turn(board, time_left)
        desperate = _is_desperate(board)
        cautious = _is_cautious(board)
        # Suppress search until we've recovered from the miss streak.
        # Iteration-6 match (6) showed desperate mode bypassing this
        # gate cost us 3 consecutive -2 misses, cascading a -21 loss.
        # Removed the desperate override — if our belief is
        # miscalibrated, being behind doesn't make -2 EV a good bet.
        suppress_search = board.turn_count < self._suppress_until_turn

        # ---- 3. Expectiminimax search ---------------------------------
        move, value, depth = self.engine.search(
            board, self.rat_tracker.belief, budget,
            desperate=desperate, cautious=cautious,
            suppress_search=suppress_search,
        )

        self.last_depth = depth
        self.last_value = value
        self.last_nodes = self.engine.nodes

        return move
