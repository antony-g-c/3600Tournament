from collections.abc import Callable
from typing import Tuple
import random
import numpy as np

from game import board, enums


class PlayerAgent:
    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        self.turn = 0

        # Rat belief
        self.T = None
        self.belief = np.ones(64, dtype=np.float64) / 64.0
        if transition_matrix is not None:
            self.T = np.array(transition_matrix, dtype=np.float64)
            self.belief = self._spawn_prior()

        # Search memory / cooldown
        self.last_player_search = None
        self.last_opponent_search = None
        self.last_failed_search_loc = None
        self.failed_search_cooldown = 0

    def commentate(self):
        return ""

    def play(
        self,
        board: board.Board,
        sensor_data: Tuple,
        time_left: Callable,
    ):
        self.turn += 1

        if self.failed_search_cooldown > 0:
            self.failed_search_cooldown -= 1

        valid_moves = board.get_valid_moves(exclude_search=False)
        if not valid_moves:
            valid_moves = board.get_valid_moves()
            return random.choice(valid_moves)

        self._handle_respawn(board)

        if self.T is not None and sensor_data is not None and len(sensor_data) == 2:
            noise, dist = sensor_data
            self._predict()
            self._update(board, noise, dist)

        scored = []
        for mv in valid_moves:
            try:
                score = self._score(board, mv)
            except Exception:
                score = -10**12
            scored.append((score, mv))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    # =========================================================
    # BELIEF
    # =========================================================

    def _spawn_prior(self):
        b = np.zeros(64, dtype=np.float64)
        b[0] = 1.0
        for _ in range(1000):
            b = b @ self.T
        s = b.sum()
        if s <= 0:
            return np.ones(64, dtype=np.float64) / 64.0
        return b / s

    def _predict(self):
        self.belief = self.belief @ self.T
        s = self.belief.sum()
        if s > 0:
            self.belief /= s
        else:
            self.belief = np.ones(64, dtype=np.float64) / 64.0

    def _update(self, board_obj, noise, est_dist):
        """
        Lightweight observation update using only the distance channel.
        This is intentionally conservative to avoid overconfident bad searches.
        """
        obs = np.zeros(64, dtype=np.float64)
        my_pos = board_obj.player_worker.position
        est_dist = int(est_dist)

        for i in range(64):
            x, y = i % 8, i // 8
            actual = abs(x - my_pos[0]) + abs(y - my_pos[1])

            p_dist = 0.0
            for off, p in [(-1, 0.12), (0, 0.70), (1, 0.12), (2, 0.06)]:
                d = max(0, actual + off)
                if d == est_dist:
                    p_dist += p

            obs[i] = p_dist

        self.belief *= obs
        s = self.belief.sum()
        if s > 0:
            self.belief /= s
        else:
            self.belief = np.ones(64, dtype=np.float64) / 64.0

    def _handle_respawn(self, board_obj):
        player_search = board_obj.player_search
        opponent_search = board_obj.opponent_search

        if (
            player_search != self.last_player_search
            or opponent_search != self.last_opponent_search
        ):
            self.last_player_search = player_search
            self.last_opponent_search = opponent_search

            # Our search happened
            if player_search[0] is not None:
                loc, success = player_search
                if success:
                    if self.T is not None:
                        self.belief = self._spawn_prior()
                    else:
                        self.belief = np.ones(64, dtype=np.float64) / 64.0
                    self.last_failed_search_loc = None
                    self.failed_search_cooldown = 0
                else:
                    self.last_failed_search_loc = loc
                    self.failed_search_cooldown = 4

            # Opponent found the rat -> reset rat prior
            if opponent_search[1]:
                if self.T is not None:
                    self.belief = self._spawn_prior()
                else:
                    self.belief = np.ones(64, dtype=np.float64) / 64.0
                self.last_failed_search_loc = None
                self.failed_search_cooldown = 0

    def _best_search_prob(self):
        return float(np.max(self.belief))

    # =========================================================
    # SCORING
    # =========================================================

    def _score(self, board_obj, mv):
        next_board = board_obj.forecast_move(mv)
        if next_board is None:
            return -10**12

        my_before = board_obj.player_worker.points
        opp_before = board_obj.opponent_worker.points
        my_after = next_board.player_worker.points
        opp_after = next_board.opponent_worker.points

        score = 0.0

        # Main objective: winning board state
        score += 15.0 * (my_after - opp_after)

        # Immediate reward
        score += 8.0 * (my_after - my_before)

        # Prefer safer point-building
        if mv.move_type == enums.MoveType.CARPET:
            score += 12.0
        elif mv.move_type == enums.MoveType.PRIME:
            score += 2.0

        # Mild center preference
        px, py = next_board.player_worker.position
        score -= 0.1 * (abs(px - 3.5) + abs(py - 3.5))

        # Search EV: extremely conservative now
        if mv.move_type == enums.MoveType.SEARCH and mv.search_loc is not None:
            x, y = mv.search_loc
            p = self.belief[y * 8 + x]
            ev = 6.0 * p - 2.0
            top_p = self._best_search_prob()

            score += 20.0 * ev

            # Strongly discourage speculative searches
            if p < 0.55:
                score -= 1000.0

            if top_p < 0.55:
                score -= 1500.0

            # Never repeat recent failed search behavior
            if self.failed_search_cooldown > 0:
                score -= 500.0

            if self.last_failed_search_loc == mv.search_loc:
                score -= 2000.0

            # When already ahead, avoid gambling
            if (my_after - opp_after) >= 2:
                score -= 300.0

        # Shallow reply lookahead
        score -= 0.4 * self._best_reply_value(next_board)

        return score

    def _best_reply_value(self, board_obj):
        rev = board_obj.get_copy()
        rev.reverse_perspective()

        replies = rev.get_valid_moves(exclude_search=False)
        if not replies:
            return 0.0

        if len(replies) > 8:
            replies = random.sample(replies, 8)

        best = 0.0
        base = rev.player_worker.points - rev.opponent_worker.points

        for r in replies:
            nb = rev.forecast_move(r)
            if nb is None:
                continue

            val = (nb.player_worker.points - nb.opponent_worker.points) - base

            if r.move_type == enums.MoveType.CARPET:
                val += 2.0
            elif r.move_type == enums.MoveType.PRIME:
                val += 0.5
            elif r.move_type == enums.MoveType.SEARCH:
                # Opponent search can swing score, but do not overvalue it
                val += 0.25

            if val > best:
                best = val

        return best



