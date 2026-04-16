from collections.abc import Callable
from typing import Tuple
import random
import numpy as np
import time

from game import board, enums


class PlayerAgent:
    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        self.turn = 0

        self.T = None
        self.belief = np.ones(64, dtype=np.float64) / 64.0
        if transition_matrix is not None:
            self.T = np.array(transition_matrix, dtype=np.float64)
            self.belief = self._spawn_prior()

        self.last_player_search = None
        self.last_opponent_search = None
        self.last_failed_search_loc = None
        self.failed_search_cooldown = 0

    def commentate(self):
        return ""

    # =========================================================
    # MAIN PLAY
    # =========================================================

    def play(self, board: board.Board, sensor_data: Tuple, time_left: Callable):
        self.turn += 1
        start = time.time()

        if self.failed_search_cooldown > 0:
            self.failed_search_cooldown -= 1

        valid_moves = board.get_valid_moves(exclude_search=False)
        if not valid_moves:
            return random.choice(board.get_valid_moves())

        self._handle_respawn(board)

        if self.T is not None and sensor_data:
            noise, dist = sensor_data
            self._predict()
            self._update(board, noise, dist)

        # 🔥 iterative deepening
        best_move = random.choice(valid_moves)
        depth = 1

        while True:
            if time.time() - start > 0.08:  # ~80ms safety
                break

            move = self._search_root(board, depth, start)
            if move is not None:
                best_move = move

            depth += 1
            if depth > 3:
                break

        return best_move

    # =========================================================
    # ROOT SEARCH
    # =========================================================

    def _search_root(self, board_obj, depth, start_time):
        moves = board_obj.get_valid_moves(exclude_search=False)
        moves = self._order_moves(board_obj, moves)

        best_val = -1e18
        best_move = None

        for mv in moves:
            if time.time() - start_time > 0.09:
                break

            if not self._valid_search(mv, self.belief):
                continue

            nb = board_obj.forecast_move(mv)
            if nb is None:
                continue

            nb.reverse_perspective()

            val = self._expectiminimax(nb, self.belief.copy(), depth - 1, "MIN")

            if val > best_val:
                best_val = val
                best_move = mv

        return best_move

    # =========================================================
    # EXPECTIMINIMAX
    # =========================================================

    def _expectiminimax(self, board_obj, belief, depth, node):
        if depth == 0:
            return self._evaluate(board_obj, belief)

        belief = self._top_k_belief(belief)

        if node == "MAX":
            return self._max_node(board_obj, belief, depth)
        elif node == "MIN":
            return self._min_node(board_obj, belief, depth)
        else:
            return self._chance_node(board_obj, belief, depth)

    def _max_node(self, board_obj, belief, depth):
        moves = board_obj.get_valid_moves(exclude_search=False)
        moves = self._order_moves(board_obj, moves)

        best = -1e18

        for mv in moves[:6]:
            if not self._valid_search(mv, belief):
                continue

            nb = board_obj.forecast_move(mv)
            if nb is None:
                continue

            nb.reverse_perspective()

            val = self._expectiminimax(nb, belief, depth - 1, "CHANCE")
            best = max(best, val)

        return best

    def _min_node(self, board_obj, belief, depth):
        moves = board_obj.get_valid_moves(exclude_search=False)
        moves = self._order_moves(board_obj, moves)

        worst = 1e18

        for mv in moves[:6]:
            nb = board_obj.forecast_move(mv)
            if nb is None:
                continue

            nb.reverse_perspective()

            val = self._expectiminimax(nb, belief, depth - 1, "CHANCE")
            worst = min(worst, val)

        return worst

    def _chance_node(self, board_obj, belief, depth):
        belief = belief @ self.T
        s = belief.sum()
        if s > 0:
            belief /= s
        else:
            belief = np.ones(64) / 64.0

        return self._expectiminimax(board_obj, belief, depth - 1, "MAX")

    # =========================================================
    # BELIEF OPTIMIZATION
    # =========================================================

    def _top_k_belief(self, belief, k=10):
        idx = np.argsort(belief)[-k:]
        new_b = np.zeros_like(belief)
        new_b[idx] = belief[idx]
        s = new_b.sum()
        return new_b / s if s > 0 else belief

    # =========================================================
    # MOVE FILTERING
    # =========================================================

    def _valid_search(self, mv, belief):
        if mv.move_type != enums.MoveType.SEARCH or mv.search_loc is None:
            return True

        x, y = mv.search_loc
        p = belief[y * 8 + x]

        if p < 0.55:
            return False
        if self.failed_search_cooldown > 0:
            return False
        if self.last_failed_search_loc == mv.search_loc:
            return False

        return True

    def _order_moves(self, board_obj, moves):
        scored = []

        for mv in moves:
            score = 0
            if mv.move_type == enums.MoveType.CARPET:
                score += 5
            elif mv.move_type == enums.MoveType.PRIME:
                score += 2
            elif mv.move_type == enums.MoveType.SEARCH:
                if mv.search_loc:
                    x, y = mv.search_loc
                    score += 20 * self.belief[y * 8 + x]

            scored.append((score, mv))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [m for _, m in scored]

    # =========================================================
    # EVALUATION
    # =========================================================

    def _evaluate(self, board_obj, belief):
        my = board_obj.player_worker.points
        opp = board_obj.opponent_worker.points

        score = 25.0 * (my - opp)

        my_pos = board_obj.player_worker.position
        expected_dist = 0.0

        for i in range(64):
            p = belief[i]
            if p < 1e-5:
                continue
            x, y = i % 8, i // 8
            d = abs(x - my_pos[0]) + abs(y - my_pos[1])
            expected_dist += p * d

        score += 10.0 / (1.0 + expected_dist)

        max_p = np.max(belief)
        score += 30.0 * max_p

        return score

    # =========================================================
    # BELIEF UPDATE (UNCHANGED CORE)
    # =========================================================

    def _spawn_prior(self):
        b = np.zeros(64)
        b[0] = 1.0
        for _ in range(1000):
            b = b @ self.T
        return b / b.sum()

    def _predict(self):
        self.belief = self.belief @ self.T
        self.belief /= self.belief.sum()

    def _update(self, board_obj, noise, est_dist):
        obs = np.zeros(64)
        my_pos = board_obj.player_worker.position

        for i in range(64):
            x, y = i % 8, i // 8
            actual = abs(x - my_pos[0]) + abs(y - my_pos[1])

            p = 0
            for off, prob in [(-1, 0.12), (0, 0.7), (1, 0.12), (2, 0.06)]:
                if max(0, actual + off) == est_dist:
                    p += prob
            obs[i] = p

        self.belief *= obs
        s = self.belief.sum()
        self.belief = self.belief / s if s > 0 else np.ones(64) / 64

    def _handle_respawn(self, board_obj):
        ps = board_obj.player_search
        os = board_obj.opponent_search

        if ps != self.last_player_search or os != self.last_opponent_search:
            self.last_player_search = ps
            self.last_opponent_search = os

            if ps[0] is not None:
                loc, success = ps
                if success:
                    self.belief = self._spawn_prior()
                    self.failed_search_cooldown = 0
                else:
                    self.last_failed_search_loc = loc
                    self.failed_search_cooldown = 4

            if os[0] is not None and not os[1]:
                x, y = os[0]
                self.belief[y * 8 + x] *= 0.01
                self.belief /= self.belief.sum()

            if os[1]:
                self.belief = self._spawn_prior()