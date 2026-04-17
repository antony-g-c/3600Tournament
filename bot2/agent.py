#CARPET_OPTIMIZED_V2
from collections import deque
from collections.abc import Callable
from typing import Tuple, Optional, Dict, List
import random
import numpy as np
import time

from game import board, enums

# Carpet payoff table (index = length)
CARPET_POINTS = {1: -1, 2: 2, 3: 4, 4: 6, 5: 10, 6: 15, 7: 21}

# Minimum carpet length worth executing (length-1 almost always loses points)
MIN_CARPET_LENGTH = 2

# How many turns from end do we switch to "endgame" mode
ENDGAME_TURNS_THRESHOLD = 15


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

        # Carpet chain tracking
        # Maps direction -> number of consecutive PRIMEs already laid from current position
        self._prime_directions: Dict = {}
        self._board_prime_map = np.zeros(64, dtype=np.int32)  # tracks primed cell ages

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

    # ─────────────────────────────────────────────
    # CARPET CHAIN ANALYSIS
    # ─────────────────────────────────────────────

    def _count_primed_chain(self, board_obj, pos, direction) -> int:
        """
        Count how many consecutive PRIMED cells exist starting from pos
        in the given direction. This tells us the potential carpet length
        if we execute a carpet move from pos.
        """
        count = 0
        current = pos
        for _ in range(7):
            current = self._step(current, direction)
            if current is None:
                break
            if not board_obj.is_valid_cell(current):
                break
            cell = board_obj.get_cell(current)
            if cell != enums.Cell.PRIMED:
                break
            # Also ensure not occupied
            if (current == board_obj.player_worker.position or
                    current == board_obj.opponent_worker.position):
                break
            count += 1
        return count

    def _best_carpet_from_pos(self, board_obj, pos) -> Tuple[int, int]:
        """
        Returns (best_carpet_length, best_carpet_points) achievable
        from a given position across all 4 directions.
        """
        best_len = 0
        best_pts = 0
        for direction in [enums.Direction.UP, enums.Direction.DOWN,
                          enums.Direction.LEFT, enums.Direction.RIGHT]:
            length = self._count_primed_chain(board_obj, pos, direction)
            if length >= MIN_CARPET_LENGTH:
                pts = CARPET_POINTS.get(length, 0)
                if pts > best_pts:
                    best_pts = pts
                    best_len = length
        return best_len, best_pts

    def _best_carpet_in_two_moves(self, board_obj) -> int:
        """
        Look ahead 2 plies to find the best carpet gain reachable.
        Considers: PRIME then CARPET, or MOVE then CARPET.
        Returns expected points gain.
        """
        my_points = board_obj.player_worker.points
        best = 0

        try:
            moves1 = board_obj.get_valid_moves(exclude_search=True)
        except Exception:
            return 0

        for mv1 in moves1:
            nb1 = board_obj.forecast_move(mv1)
            if nb1 is None:
                continue

            try:
                moves2 = nb1.get_valid_moves(exclude_search=True)
            except Exception:
                continue

            for mv2 in moves2:
                if mv2.move_type != enums.MoveType.CARPET:
                    continue
                nb2 = nb1.forecast_move(mv2)
                if nb2 is None:
                    continue
                gain = nb2.player_worker.points - my_points
                if gain > best:
                    best = gain

        return best

    def _carpet_potential_score(self, board_obj) -> float:
        """
        Evaluate the carpet-building potential of the current board state.
        Rewards:
        - Long chains of PRIMED cells reachable from current pos
        - Chains that are one PRIME away from a high-value carpet
        Uses the convex payoff structure: length^1.8 approximates the reward curve.
        """
        pos = board_obj.player_worker.position
        score = 0.0

        for direction in [enums.Direction.UP, enums.Direction.DOWN,
                          enums.Direction.LEFT, enums.Direction.RIGHT]:
            chain_len = self._count_primed_chain(board_obj, pos, direction)

            if chain_len == 0:
                continue

            pts = CARPET_POINTS.get(chain_len, 0)
            if pts > 0:
                # Immediate carpet value — weighted by how good the payoff is
                score += pts * 1.2
            elif pts < 0:
                # Active penalty for length-1 chains (don't rush to cash them)
                score -= 2.0

            # One more PRIME would give this carpet
            extended = CARPET_POINTS.get(chain_len + 1, 0)
            if extended > pts + 2:
                # There's a meaningful gain from waiting one more turn
                score += (extended - pts) * 0.6

        return score

    def _primed_cell_count(self, board_obj) -> int:
        """Count total PRIMED cells on board (proxy for setup investment)."""
        count = 0
        for y in range(8):
            for x in range(8):
                if board_obj.get_cell((x, y)) == enums.Cell.PRIMED:
                    count += 1
        return count

    # ─────────────────────────────────────────────
    # GAME PHASE DETECTION
    # ─────────────────────────────────────────────

    def _turns_remaining(self, board_obj) -> int:
        return board_obj.player_worker.turns_left

    def _is_endgame(self, board_obj) -> bool:
        return self._turns_remaining(board_obj) <= ENDGAME_TURNS_THRESHOLD

    def _is_desperate(self, board_obj) -> bool:
        """Losing significantly with few turns left."""
        lead = board_obj.player_worker.points - board_obj.opponent_worker.points
        return lead <= -4 and self._turns_remaining(board_obj) <= 20

    # ─────────────────────────────────────────────
    # SEARCH TREE
    # ─────────────────────────────────────────────

    def _search_root(self, board_obj, depth, start_time):
        moves = board_obj.get_valid_moves(exclude_search=False)
        moves = self._order_moves(board_obj, moves)

        best_val = -1e18
        best_move = None

        for mv in moves:
            if time.time() - start_time > 0.09:
                break

            # Block bad carpets at root
            if not self._carpet_is_acceptable(board_obj, mv):
                continue

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

        for mv in moves[:8]:
            if not self._carpet_is_acceptable(board_obj, mv):
                continue
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

        for mv in moves[:8]:
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

    # ─────────────────────────────────────────────
    # CARPET ACCEPTABILITY GATE
    # ─────────────────────────────────────────────

    def _carpet_is_acceptable(self, board_obj, mv) -> bool:
        """
        Hard gate: block carpet moves that are net-negative or premature.
        
        Rules:
        1. Never take a length-1 carpet (costs 1 point) unless it's endgame
           desperation with no alternatives.
        2. Don't cash a length-2 carpet (2 pts) if a length-4+ is reachable
           within 2 moves.
        3. Don't cash any carpet if extending by 1 PRIME would reach a 
           significantly better payoff (unless endgame).
        """
        if mv.move_type != enums.MoveType.CARPET:
            return True

        length = mv.roll_length
        pts = CARPET_POINTS.get(length, 0)
        endgame = self._is_endgame(board_obj)

        # Rule 1: Never length-1 unless desperate endgame
        if length == 1:
            if endgame and self._is_desperate(board_obj):
                return True
            return False

        # Rule 2: Length-2 only if nothing better is reachable soon
        if length == 2 and not endgame:
            future = self._best_carpet_in_two_moves(board_obj)
            if future >= 6:
                return False

        # Rule 3: If extending by 1 gives >3 extra points, wait
        # (unless we're in endgame where turns are precious)
        if not endgame:
            extended_pts = CARPET_POINTS.get(length + 1, pts)
            gain_from_waiting = extended_pts - pts
            if gain_from_waiting >= 4:
                return False

        return True

    # ─────────────────────────────────────────────
    # MOVE ORDERING
    # ─────────────────────────────────────────────

    def _order_moves(self, board_obj, moves):
        scored = []
        my_points = board_obj.player_worker.points
        lead = board_obj.player_worker.points - board_obj.opponent_worker.points
        endgame = self._is_endgame(board_obj)
        turns_left = self._turns_remaining(board_obj)
        current_rat_dist = self._best_belief_distance(board_obj.player_worker.position)

        # Pre-compute two-move carpet ceiling for context
        two_move_carpet = self._best_carpet_in_two_moves(board_obj)

        for mv in moves:
            score = 0.0
            immediate_gain = 0

            nb = board_obj.forecast_move(mv)
            if nb is not None:
                immediate_gain = nb.player_worker.points - my_points
                score += 7.0 * immediate_gain

            if mv.move_type == enums.MoveType.CARPET:
                length = mv.roll_length
                pts = CARPET_POINTS.get(length, 0)

                if pts < 0:
                    # Length-1: penalize heavily
                    score -= 30.0
                elif length <= 2 and two_move_carpet >= 6 and not endgame:
                    # Better carpet coming — deprioritize
                    score -= 10.0
                else:
                    # Reward longer carpets more aggressively (reflects convex payoff)
                    score += 10.0 + pts * 1.8

                    # Endgame: bonus for cashing NOW
                    if endgame:
                        score += pts * 0.5 * (ENDGAME_TURNS_THRESHOLD / max(1, turns_left))

            elif mv.move_type == enums.MoveType.PRIME:
                # Value the PRIME by what carpet it sets up
                pos_after = None
                if nb is not None:
                    pos_after = nb.player_worker.position

                best_chain_pts = 0
                if pos_after is not None:
                    for direction in [enums.Direction.UP, enums.Direction.DOWN,
                                      enums.Direction.LEFT, enums.Direction.RIGHT]:
                        chain = self._count_primed_chain(nb, pos_after, direction)
                        if chain >= 1:
                            # This PRIME extends an existing chain — very valuable
                            expected_pts = CARPET_POINTS.get(chain + 1, 0)
                            best_chain_pts = max(best_chain_pts, expected_pts)

                score += 3.0 + best_chain_pts * 0.9

                if not endgame:
                    # Look-ahead: how good could this chain get?
                    future = self._best_carpet_in_two_moves(nb) if nb else 0
                    score += future * 0.5

            elif mv.move_type == enums.MoveType.SEARCH and mv.search_loc:
                x, y = mv.search_loc
                p = float(self.belief[y * 8 + x])
                score += 22.0 * p - 6.0

                # Desperate rat-finding bonus
                if lead <= -3:
                    score += 12.0 * p
                elif lead <= 1:
                    score += 6.0 * p
                elif lead >= 3 and not endgame:
                    score -= 2.0 * p

            else:
                # Plain move: value by rat proximity + carpet setup
                score += 0.5
                if nb is not None and immediate_gain == 0:
                    next_rat_dist = self._best_belief_distance(nb.player_worker.position)
                    rat_progress = current_rat_dist - next_rat_dist
                    score -= 2.0
                    score += 2.0 * rat_progress

                    # Also reward moving toward high-value carpet chain positions
                    if nb is not None:
                        carpet_pot = self._carpet_potential_score(nb)
                        score += carpet_pot * 0.4

            new_pos = self._move_destination(board_obj, mv)
            if new_pos is not None:
                score += self._novelty_bonus(new_pos)

            scored.append((score, mv))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [m for _, m in scored]

    def _pick_safe_default(self, board_obj, valid_moves):
        ordered = self._order_moves(board_obj, valid_moves)
        for mv in ordered:
            if not self._carpet_is_acceptable(board_obj, mv):
                continue
            if self._valid_search(board_obj, mv, self.belief):
                return mv
        # Fall back to first acceptable carpet or any move
        for mv in ordered:
            if self._carpet_is_acceptable(board_obj, mv):
                return mv
        return ordered[0] if ordered else random.choice(valid_moves)

    # ─────────────────────────────────────────────
    # EVALUATION FUNCTION
    # ─────────────────────────────────────────────

    def _evaluate(self, board_obj, belief):
        my = board_obj.player_worker.points
        opp = board_obj.opponent_worker.points
        my_pos = board_obj.player_worker.position
        opp_pos = board_obj.opponent_worker.position
        turns_left = self._turns_remaining(board_obj)
        lead = my - opp
        endgame = self._is_endgame(board_obj)

        # ── Point lead (scales with game phase) ──
        # Late game: a 1-point lead is much more valuable
        phase_multiplier = 28.0 + (30.0 * max(0, ENDGAME_TURNS_THRESHOLD - turns_left) / ENDGAME_TURNS_THRESHOLD)
        score = phase_multiplier * lead

        # ── Rat proximity ──
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

        # ── Race value ──
        score += 1.6 * self._race_value(my_pos, opp_pos, belief)

        # ── Mobility ──
        score += 2.5 * self._mobility_value(board_obj)

        # ── Anti-loop ──
        score += 3.0 * self._anti_loop_eval(my_pos)

        # ── Carpet chain potential (NEW) ──
        carpet_pot = self._carpet_potential_score(board_obj)
        # Weight carpet potential less in endgame (no time to build)
        carpet_weight = 4.5 if not endgame else 1.5
        score += carpet_weight * carpet_pot

        # ── Two-move carpet ceiling (NEW) ──
        # Reward board states where a big carpet is imminent
        two_move = self._best_carpet_in_two_moves(board_obj)
        score += 2.2 * two_move

        # ── Desperate mode: weight rat finding much more ──
        if self._is_desperate(board_obj):
            score += 8.0 * max_p
            score += 4.0 / (1.0 + expected_dist)

        return score

    # ─────────────────────────────────────────────
    # SEARCH VALIDITY
    # ─────────────────────────────────────────────

    def _valid_search(self, board_obj, mv, belief):
        if mv.move_type != enums.MoveType.SEARCH or mv.search_loc is None:
            return True

        x, y = mv.search_loc
        p = belief[y * 8 + x]
        lead = board_obj.player_worker.points - board_obj.opponent_worker.points
        my_pos = board_obj.player_worker.position
        dist = abs(x - my_pos[0]) + abs(y - my_pos[1])
        top_p = float(np.max(belief))
        turns_left = self._turns_remaining(board_obj)

        # Dynamic threshold: decays with urgency
        # More turns left = higher bar (don't waste turns on weak leads)
        # Fewer turns left = lower bar (must act)
        urgency = max(0.0, min(1.0, turns_left / ENDGAME_TURNS_THRESHOLD))
        base_threshold = 0.64

        if lead <= -4:
            base_threshold = 0.44
        elif lead <= -2:
            base_threshold = 0.50
        elif lead <= 1:
            base_threshold = 0.54
        elif 2 <= lead <= 3:
            base_threshold = 0.62
        elif lead >= 4:
            base_threshold = 0.69

        # Scale threshold toward 0.40 in late game desperation
        threshold = base_threshold * urgency + 0.40 * (1.0 - urgency)

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

        if self._has_strong_carpet_move(board_obj, lead, p, top_p):
            return False

        return True

    def _has_strong_carpet_move(self, board_obj, lead, p, top_p) -> bool:
        """
        Don't search if there's a high-value carpet move available.
        Extended from original to use actual carpet payoffs.
        """
        if lead < 2:
            return False

        moves = board_obj.get_valid_moves(exclude_search=True)
        my_points = board_obj.player_worker.points
        best_carpet_gain = 0

        for mv in moves:
            if mv.move_type != enums.MoveType.CARPET:
                continue
            length = mv.roll_length
            pts = CARPET_POINTS.get(length, 0)
            if pts <= 0:
                continue
            nb = board_obj.forecast_move(mv)
            if nb is None:
                continue
            gain = nb.player_worker.points - my_points
            best_carpet_gain = max(best_carpet_gain, gain)

        # Thresholds scale with lead size
        if lead >= 4 and best_carpet_gain >= 4 and p < max(0.66, top_p - 0.02):
            return True
        if lead >= 2 and best_carpet_gain >= 6 and p < 0.72:
            return True
        if lead >= 3 and best_carpet_gain >= 10 and p < 0.80:
            return True

        return False

    # ─────────────────────────────────────────────
    # BELIEF TRACKING
    # ─────────────────────────────────────────────

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

    # ─────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────

    def _step(self, pos, direction) -> Optional[Tuple[int, int]]:
        x, y = pos
        if direction == enums.Direction.UP:
            y -= 1
        elif direction == enums.Direction.DOWN:
            y += 1
        elif direction == enums.Direction.LEFT:
            x -= 1
        elif direction == enums.Direction.RIGHT:
            x += 1
        if 0 <= x < 8 and 0 <= y < 8:
            return (x, y)
        return None

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

    def _move_destination(self, board_obj, mv) -> Optional[Tuple[int, int]]:
        if mv.move_type == enums.MoveType.SEARCH:
            return board_obj.player_worker.position
        nb = board_obj.forecast_move(mv)
        if nb is None:
            return None
        return nb.player_worker.position

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