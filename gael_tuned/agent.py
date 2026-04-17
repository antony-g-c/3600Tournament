#GAEL_TUNED
from collections import deque
from collections.abc import Callable
from typing import Tuple, Optional
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

        self.position_history = deque(maxlen=8)
        self.search_history = deque(maxlen=4)

    def commentate(self):
        return ""

    def play(self, board: board.Board, sensor_data: Tuple, time_left: Callable):
        self.turn += 1
        start = time.time()

        if self.failed_search_cooldown > 0:
            self.failed_search_cooldown -= 1

        my_pos = board.player_worker.position
        self.position_history.append(my_pos)

        valid_moves = board.get_valid_moves(exclude_search=False)
        if not valid_moves:
            fallback = board.get_valid_moves()
            return random.choice(fallback) if fallback else random.choice(valid_moves)

        self._handle_respawn(board)

        if self.T is not None and sensor_data:
            noise, dist = sensor_data
            self._predict()
            self._update(board, noise, dist)

        best_move = self._pick_safe_default(board, valid_moves)
        depth = 1

        while True:
            elapsed = time.time() - start
            if elapsed > 0.085:
                break

            move = self._search_root(board, depth, start)
            if move is not None:
                best_move = move

            depth += 1
            if depth > 3:
                break

        if best_move.move_type == enums.MoveType.SEARCH:
            self.search_history.append(1)
        else:
            self.search_history.append(0)

        return best_move

    def _search_root(self, board_obj, depth, start_time):
        moves = board_obj.get_valid_moves(exclude_search=False)
        moves = self._order_moves(board_obj, moves)

        best_val = -1e18
        best_move = None

        for mv in moves:
            if time.time() - start_time > 0.09:
                break

            if not self._valid_search(board_obj, mv, self.belief):
                continue

            nb = board_obj.forecast_move(mv)
            if nb is None:
                continue

            nb.reverse_perspective()
            val = self._expectiminimax(nb, self.belief.copy(), depth - 1, "MIN", start_time)

            if val > best_val:
                best_val = val
                best_move = mv

        return best_move

    def _expectiminimax(self, board_obj, belief, depth, node, start_time):
        if time.time() - start_time > 0.092:
            return self._evaluate(board_obj, belief)

        if depth == 0:
            return self._evaluate(board_obj, belief)

        belief = self._top_k_belief(belief)

        if node == "MAX":
            return self._max_node(board_obj, belief, depth, start_time)
        elif node == "MIN":
            return self._min_node(board_obj, belief, depth, start_time)
        else:
            return self._chance_node(board_obj, belief, depth, start_time)

    def _max_node(self, board_obj, belief, depth, start_time):
        moves = board_obj.get_valid_moves(exclude_search=False)
        moves = self._order_moves(board_obj, moves)

        best = -1e18

        for mv in moves[:7]:
            if not self._valid_search(board_obj, mv, belief):
                continue

            nb = board_obj.forecast_move(mv)
            if nb is None:
                continue

            nb.reverse_perspective()
            val = self._expectiminimax(nb, belief, depth - 1, "CHANCE", start_time)
            if val > best:
                best = val

        return best if best > -1e17 else self._evaluate(board_obj, belief)

    def _min_node(self, board_obj, belief, depth, start_time):
        moves = board_obj.get_valid_moves(exclude_search=False)
        moves = self._order_moves(board_obj, moves)

        worst = 1e18

        for mv in moves[:7]:
            nb = board_obj.forecast_move(mv)
            if nb is None:
                continue

            nb.reverse_perspective()
            val = self._expectiminimax(nb, belief, depth - 1, "CHANCE", start_time)
            if val < worst:
                worst = val

        return worst if worst < 1e17 else self._evaluate(board_obj, belief)

    def _chance_node(self, board_obj, belief, depth, start_time):
        if self.T is None:
            return self._expectiminimax(board_obj, belief, depth - 1, "MAX", start_time)

        belief = belief @ self.T
        s = belief.sum()
        if s > 0:
            belief /= s
        else:
            belief = np.ones(64, dtype=np.float64) / 64.0

        return self._expectiminimax(board_obj, belief, depth - 1, "MAX", start_time)

    def _top_k_belief(self, belief, k=10):
        idx = np.argsort(belief)[-k:]
        new_b = np.zeros_like(belief)
        new_b[idx] = belief[idx]
        s = new_b.sum()
        return new_b / s if s > 0 else belief

    def _spawn_prior(self):
        if self.T is None:
            return np.ones(64, dtype=np.float64) / 64.0

        b = np.zeros(64, dtype=np.float64)
        b[0] = 1.0
        for _ in range(1000):
            b = b @ self.T
        s = b.sum()
        return b / s if s > 0 else np.ones(64, dtype=np.float64) / 64.0

    def _predict(self):
        if self.T is None:
            return
        self.belief = self.belief @ self.T
        s = self.belief.sum()
        self.belief = self.belief / s if s > 0 else np.ones(64, dtype=np.float64) / 64.0

    def _update(self, board_obj, noise, est_dist):
        obs = np.zeros(64, dtype=np.float64)
        my_pos = board_obj.player_worker.position

        for i in range(64):
            x, y = i % 8, i // 8
            actual = abs(x - my_pos[0]) + abs(y - my_pos[1])

            p = 0.0
            for off, prob in [(-1, 0.12), (0, 0.70), (1, 0.12), (2, 0.06)]:
                if max(0, actual + off) == est_dist:
                    p += prob
            obs[i] = p

        self.belief *= obs
        s = self.belief.sum()
        if s > 0:
            self.belief /= s
        else:
            self.belief = np.ones(64, dtype=np.float64) / 64.0

    def _valid_search(self, board_obj, mv, belief):
        if mv.move_type != enums.MoveType.SEARCH or mv.search_loc is None:
            return True

        x, y = mv.search_loc
        p = belief[y * 8 + x]
        lead = board_obj.player_worker.points - board_obj.opponent_worker.points
        my_pos = board_obj.player_worker.position
        dist = abs(x - my_pos[0]) + abs(y - my_pos[1])
        top_p = float(np.max(belief))

        threshold = 0.64
        if lead <= -3:
            threshold = 0.50
        elif lead <= 0:
            threshold = 0.58
        elif 1 <= lead <= 3:
            threshold = 0.64
        elif lead >= 4:
            threshold = 0.69

        if dist >= 3 and lead >= 0:
            threshold += 0.03

        if p < threshold:
            return False
        if self.failed_search_cooldown > 0:
            return False
        if self.last_failed_search_loc == mv.search_loc:
            return False

        recent_searches = sum(self.search_history)
        if recent_searches >= 2 and p < 0.76 and lead >= 0:
            return False

        if self._has_strong_scoring_move(board_obj, lead, p, top_p):
            return False

        return True

    def _has_strong_scoring_move(self, board_obj, lead, p, top_p):
        moves = board_obj.get_valid_moves(exclude_search=True)
        my_points = board_obj.player_worker.points
        best_carpet_gain = 0
        best_prime_gain = 0

        for mv in moves:
            nb = board_obj.forecast_move(mv)
            if nb is None:
                continue

            gain = nb.player_worker.points - my_points
            if mv.move_type == enums.MoveType.CARPET:
                best_carpet_gain = max(best_carpet_gain, gain)
            elif mv.move_type == enums.MoveType.PRIME:
                best_prime_gain = max(best_prime_gain, gain)

        if lead >= 1:
            if best_carpet_gain >= 4 and p < max(0.66, top_p - 0.02):
                return True
            if best_prime_gain >= 1 and p < 0.70:
                return True

        if lead >= 3:
            if best_carpet_gain >= 3 and p < max(0.70, top_p - 0.01):
                return True

        return False

    def _order_moves(self, board_obj, moves):
        scored = []
        my_points = board_obj.player_worker.points
        lead = board_obj.player_worker.points - board_obj.opponent_worker.points
        current_rat_dist = self._best_belief_distance(board_obj.player_worker.position)

        for mv in moves:
            score = 0.0
            immediate_gain = 0
            future_carpet_gain = 0

            nb = board_obj.forecast_move(mv)
            if nb is not None:
                immediate_gain = nb.player_worker.points - my_points
                score += 6.0 * immediate_gain
                future_carpet_gain = self._best_future_carpet_gain(nb)
                score += 1.8 * future_carpet_gain

            if mv.move_type == enums.MoveType.CARPET:
                score += 12.0
                # Do not cash tiny carpets too early if a much longer one is one move away.
                if immediate_gain <= 2 and future_carpet_gain >= 4:
                    score -= 6.0
            elif mv.move_type == enums.MoveType.PRIME:
                score += 4.0
            elif mv.move_type == enums.MoveType.SEARCH and mv.search_loc:
                x, y = mv.search_loc
                p = float(self.belief[y * 8 + x])
                score += 20.0 * p
                score -= 6.0
                if lead <= 0:
                    score += 4.0 * p
                if lead <= -3:
                    score += 8.0 * p
                if 1 <= lead <= 3:
                    score -= 1.0 * p
            else:
                score += 0.5
                if nb is not None and immediate_gain == 0:
                    next_rat_dist = self._best_belief_distance(nb.player_worker.position)
                    rat_progress = current_rat_dist - next_rat_dist

                    score -= 2.0
                    score += 1.8 * rat_progress
                    score += 1.2 * max(0, future_carpet_gain - 2)
                    if rat_progress <= 0 and future_carpet_gain <= 2:
                        score -= 3.5

            new_pos = self._move_destination(board_obj, mv)
            if new_pos is not None:
                score += self._novelty_bonus(new_pos)

            scored.append((score, mv))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [m for _, m in scored]

    def _pick_safe_default(self, board_obj, valid_moves):
        ordered = self._order_moves(board_obj, valid_moves)
        for mv in ordered:
            if self._valid_search(board_obj, mv, self.belief):
                return mv
        return ordered[0] if ordered else random.choice(valid_moves)

    def _evaluate(self, board_obj, belief):
        my = board_obj.player_worker.points
        opp = board_obj.opponent_worker.points
        my_pos = board_obj.player_worker.position
        opp_pos = board_obj.opponent_worker.position

        score = 28.0 * (my - opp)

        expected_dist = 0.0
        for i in range(64):
            p = belief[i]
            if p < 1e-6:
                continue
            x, y = i % 8, i // 8
            d = abs(x - my_pos[0]) + abs(y - my_pos[1])
            expected_dist += p * d

        score += 8.0 / (1.0 + expected_dist)

        max_p = float(np.max(belief))
        score += 18.0 * max_p
        score += 1.6 * self._race_value(my_pos, opp_pos, belief)
        score += 2.5 * self._mobility_value(board_obj)
        score += 3.0 * self._anti_loop_eval(my_pos)

        if (my - opp) <= -2:
            score += 5.0 * max_p

        return score

    def _race_value(self, my_pos, opp_pos, belief):
        idx = np.argsort(belief)[-6:]
        val = 0.0
        for i in idx:
            p = belief[i]
            x, y = i % 8, i // 8
            d_me = abs(x - my_pos[0]) + abs(y - my_pos[1])
            d_opp = abs(x - opp_pos[0]) + abs(y - opp_pos[1])
            val += p * (d_opp - d_me)
        return val

    def _mobility_value(self, board_obj):
        try:
            my_moves = len(board_obj.get_valid_moves(exclude_search=True))
            return min(my_moves, 10)
        except Exception:
            return 0.0

    def _best_future_carpet_gain(self, board_obj):
        my_points = board_obj.player_worker.points
        best_gain = 0
        try:
            moves = board_obj.get_valid_moves(exclude_search=True)
        except Exception:
            return 0

        for mv in moves:
            if mv.move_type != enums.MoveType.CARPET:
                continue
            nb = board_obj.forecast_move(mv)
            if nb is None:
                continue
            gain = nb.player_worker.points - my_points
            if gain > best_gain:
                best_gain = gain

        return best_gain

    def _best_belief_distance(self, pos):
        idx = int(np.argmax(self.belief))
        target = (idx % 8, idx // 8)
        return abs(target[0] - pos[0]) + abs(target[1] - pos[1])

    def _anti_loop_eval(self, my_pos):
        if not self.position_history:
            return 0.0

        repeats = sum(1 for p in self.position_history if p == my_pos)
        return -2.5 * max(0, repeats - 2)

    def _novelty_bonus(self, pos):
        count = sum(1 for p in self.position_history if p == pos)
        if count == 0:
            return 2.5
        if count == 1:
            return 0.8
        if count >= 3:
            return -4.0
        return -1.5

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
                    self.last_failed_search_loc = None
                else:
                    self.last_failed_search_loc = loc
                    self.failed_search_cooldown = 4

            if os[0] is not None and not os[1]:
                x, y = os[0]
                idx = y * 8 + x
                self.belief[idx] *= 0.01
                s = self.belief.sum()
                self.belief = self.belief / s if s > 0 else np.ones(64, dtype=np.float64) / 64.0

            if os[0] is not None and os[1]:
                self.belief = self._spawn_prior()

    def _move_destination(self, board_obj, mv) -> Optional[Tuple[int, int]]:
        if mv.move_type == enums.MoveType.SEARCH:
            return board_obj.player_worker.position

        nb = board_obj.forecast_move(mv)
        if nb is None:
            return None
        return nb.player_worker.position
