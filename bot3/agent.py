
from collections import deque
from collections.abc import Callable
from typing import Tuple, Optional, List
import random
import numpy as np
import time

from game import board, enums

# ── Carpet payoff (matches rulebook exactly) ─────────────────────────────────
CARPET_PTS = {1: -1, 2: 2, 3: 4, 4: 6, 5: 10, 6: 15, 7: 21}

# Noise emission probabilities (Cell → (squeak, scratch, squeal))
NOISE_PROBS = {
    enums.Cell.BLOCKED: (0.5,  0.3,  0.2),
    enums.Cell.SPACE:   (0.7,  0.15, 0.15),
    enums.Cell.PRIMED:  (0.1,  0.8,  0.1),
    enums.Cell.CARPET:  (0.1,  0.1,  0.8),
}

ENDGAME_TURNS = 15   # turns remaining at which "endgame" begins
BOARD = 8


def _cp(n: int) -> int:
    """Carpet points for roll length n."""
    return CARPET_PTS.get(min(n, 7), 21)


class PlayerAgent:

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        self.turn = 0

        self.T  = None
        self.T2 = None
        self._spawn_prior_cache = None

        if transition_matrix is not None:
            self.T  = np.array(transition_matrix, dtype=np.float64)
            self.T2 = self.T @ self.T
            self.belief = self._compute_spawn_prior()
        else:
            self.belief = np.ones(64, dtype=np.float64) / 64.0

        # Search memory
        self.last_player_search      = None
        self.last_opponent_search    = None
        self.last_failed_search_loc  = None
        self.failed_search_cooldown  = 0

        # History
        self.position_history       = deque(maxlen=8)
        self.search_history         = deque(maxlen=4)
        self.obs_likelihood_history = deque(maxlen=2)   # from Agent 4

    def commentate(self):
        return ""

    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN ENTRY POINT
    # ═══════════════════════════════════════════════════════════════════════════

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

        # Safe default via move ordering (no tree search)
        best_move = self._pick_safe_default(board, valid_moves)

        # Iterative-deepening expectiminimax
        for depth in range(1, 4):
            if time.time() - start > 0.085:
                break
            mv = self._search_root(board, depth, start)
            if mv is not None:
                best_move = mv

        self.search_history.append(
            1 if best_move.move_type == enums.MoveType.SEARCH else 0
        )
        return best_move

    # ═══════════════════════════════════════════════════════════════════════════
    # EXPECTIMINIMAX TREE
    # ═══════════════════════════════════════════════════════════════════════════

    def _search_root(self, board_obj, depth, start_time):
        moves = self._order_moves(board_obj, board_obj.get_valid_moves(exclude_search=False))

        best_val  = -1e18
        best_move = None

        for mv in moves:
            if time.time() - start_time > 0.09:
                break
            if not self._carpet_is_acceptable(board_obj, mv):
                continue
            if not self._valid_search(board_obj, mv, self.belief):
                continue

            nb = board_obj.forecast_move(mv)
            if nb is None:
                continue
            nb.reverse_perspective()

            val = self._minimax(nb, self.belief.copy(), depth - 1, "MIN", start_time)
            if val > best_val:
                best_val  = val
                best_move = mv

        return best_move

    def _minimax(self, board_obj, belief, depth, node, start_time):
        if time.time() - start_time > 0.092 or depth == 0:
            return self._evaluate(board_obj, belief)

        belief = self._top_k_belief(belief)

        if node == "MAX":
            moves = self._order_moves(board_obj, board_obj.get_valid_moves(exclude_search=False))
            best  = -1e18
            for mv in moves[:8]:
                if not self._carpet_is_acceptable(board_obj, mv):
                    continue
                if not self._valid_search(board_obj, mv, belief):
                    continue
                nb = board_obj.forecast_move(mv)
                if nb is None:
                    continue
                nb.reverse_perspective()
                val = self._minimax(nb, belief, depth - 1, "CHANCE", start_time)
                if val > best:
                    best = val
            return best if best > -1e17 else self._evaluate(board_obj, belief)

        elif node == "MIN":
            moves = self._order_moves(board_obj, board_obj.get_valid_moves(exclude_search=False))
            worst = 1e18
            for mv in moves[:8]:
                nb = board_obj.forecast_move(mv)
                if nb is None:
                    continue
                nb.reverse_perspective()
                val = self._minimax(nb, belief, depth - 1, "CHANCE", start_time)
                if val < worst:
                    worst = val
            return worst if worst < 1e17 else self._evaluate(board_obj, belief)

        else:  # CHANCE — propagate belief through rat's transition
            if self.T is None:
                return self._minimax(board_obj, belief, depth - 1, "MAX", start_time)
            new_belief = belief @ self.T
            s = new_belief.sum()
            new_belief = new_belief / s if s > 0 else np.ones(64) / 64.0
            return self._minimax(board_obj, new_belief, depth - 1, "MAX", start_time)

    # ═══════════════════════════════════════════════════════════════════════════
    # STATIC EVALUATION
    # ═══════════════════════════════════════════════════════════════════════════

    def _evaluate(self, board_obj, belief) -> float:
        my      = board_obj.player_worker.points
        opp     = board_obj.opponent_worker.points
        my_pos  = board_obj.player_worker.position
        opp_pos = board_obj.opponent_worker.position
        turns   = board_obj.player_worker.turns_left
        lead    = my - opp
        endgame = turns <= ENDGAME_TURNS

        # ── 1. Point lead — scales up in endgame ─────────────────────────────
        # Each point is worth more when fewer turns remain to recover.
        phase_mult = 28.0 + 22.0 * max(0, (ENDGAME_TURNS - turns) / ENDGAME_TURNS)
        score = phase_mult * lead

        # ── 2. Rat proximity (expected Manhattan distance) ────────────────────
        expected_dist = float(np.dot(
            belief,
            [abs(i % BOARD - my_pos[0]) + abs(i // BOARD - my_pos[1]) for i in range(64)]
        ))
        score += 8.0 / (1.0 + expected_dist)

        # ── 3. Peak belief — high concentration means imminent capture ────────
        max_p  = float(np.max(belief))
        score += 18.0 * max_p

        # ── 4. Race: do we out-distance the opponent to the rat? ──────────────
        score += 1.6 * self._race_value(my_pos, opp_pos, belief)

        # ── 5. Mobility — more options is better ─────────────────────────────
        score += 2.5 * self._mobility_value(board_obj)

        # ── 6. Anti-loop ──────────────────────────────────────────────────────
        score += 3.0 * self._anti_loop_eval(my_pos)

        # ── 7. Carpet chain potential (phase-weighted) ────────────────────────
        carpet_pot = self._carpet_potential_score(board_obj)
        score += (4.5 if not endgame else 1.5) * carpet_pot

        # ── 8. Two-move carpet ceiling ────────────────────────────────────────
        two_move = self._best_carpet_in_two_moves(board_obj)
        score += 2.5 * two_move

        # ── 9. Corridor potential (long unblocked lines) ──────────────────────
        corr = self._corridor_profile(board_obj, my_pos)
        score += 2.0 * corr["value"]
        if corr["best_run"] >= 4:
            score += 6.0
        if corr["best_run"] >= 5:
            score += 5.0

        # ── 10. Desperation mode ──────────────────────────────────────────────
        if self._is_desperate(board_obj):
            score += 8.0 * max_p
            score += 4.0 / (1.0 + expected_dist)

        return score

    # ═══════════════════════════════════════════════════════════════════════════
    # MOVE ORDERING  (unified, phase-aware, carpet-safe)
    # ═══════════════════════════════════════════════════════════════════════════

    def _order_moves(self, board_obj, moves) -> list:
        """
        Score every move and return sorted best-first.

        Key design principles
        ─────────────────────
        • CARPET: score = actual payoff × phase multiplier, capped so that
          roll=1 (−1 pt) is always dead-last. Long rolls get convex bonuses.
        • PRIME:  score = 1 pt immediate + chain-extension value + setup look-ahead.
          We compute how many consecutive primed cells the new square creates
          and weight by the payoff that chain would unlock.
        • SEARCH: score = true EV (6P−2) × 10, adjusted for lead/urgency.
        • PLAIN:  score = rat-proximity delta + carpet-setup potential.

        All four types compete on the same scale so the sort is meaningful.
        """
        scored     = []
        my_pts     = board_obj.player_worker.points
        lead       = my_pts - board_obj.opponent_worker.points
        endgame    = self._is_endgame(board_obj)
        turns      = board_obj.player_worker.turns_left
        early_game = self.turn <= 20
        my_pos     = board_obj.player_worker.position

        rat_dist     = self._best_belief_distance(my_pos)
        cur_profile  = self._carpet_profile(board_obj)
        cur_corr     = self._corridor_profile(board_obj, my_pos)
        two_move_cap = self._best_carpet_in_two_moves(board_obj)

        for mv in moves:
            score = 0.0
            nb    = board_obj.forecast_move(mv)
            imm   = (nb.player_worker.points - my_pts) if nb is not None else 0

            # Immediate gain on the same scale for all move types
            score += 7.0 * imm

            # ── Per-type scoring ──────────────────────────────────────────────

            if mv.move_type == enums.MoveType.CARPET:
                length = mv.roll_length
                pts    = _cp(length)

                if pts < 0:
                    # roll=1: ALWAYS dead-last regardless of anything else
                    score -= 50.0
                else:
                    # Convex payoff bonus: longer rolls are disproportionately good
                    score += 8.0 + pts * 1.8

                    # Endgame: urgency bonus scales with how late it is
                    if endgame:
                        urgency = max(0.0, 1.0 - turns / ENDGAME_TURNS)
                        score  += pts * 1.5 * urgency

                    # Anti-premature: penalise if a much better roll is ≤2 moves away
                    if not endgame and two_move_cap - pts >= 6:
                        score -= 12.0

                    # Line-loss penalty: how much carpet potential do we destroy?
                    if nb is not None:
                        next_profile = self._carpet_profile(nb)
                        line_loss    = max(0.0, cur_profile["line_value"] - next_profile["line_value"])
                        score       -= 1.4 * line_loss

                    # Flat reward tiers for genuinely good rolls
                    if pts >= 4:
                        score += 4.0
                    if pts >= 6:
                        score += 8.0
                    if pts >= 10:
                        score += 12.0

            elif mv.move_type == enums.MoveType.PRIME:
                # 1 point immediate + chain-extension value
                score += 3.0   # base (1 pt captured in imm already)

                if nb is not None:
                    nb_pos = nb.player_worker.position
                    # Find the longest chain this PRIME creates or extends
                    best_chain_pts = 0
                    for direction in enums.Direction:
                        chain = self._count_primed_chain(nb, nb_pos, direction)
                        if chain >= 1:
                            # After one more PRIME/CARPET, we'd have chain+1
                            chain_pts = _cp(chain + 1)
                            best_chain_pts = max(best_chain_pts, chain_pts)
                    score += best_chain_pts * 0.85

                    # Setup look-ahead (from Agent 4)
                    if early_game or not endgame:
                        setup = self._setup_plan_profile(nb)
                        score += 1.2 * setup["value"]
                        if setup["best_roll"] >= 4:
                            score += 14.0
                        if setup["best_roll"] >= 5:
                            score += 10.0

                    # Corridor improvement
                    next_corr    = self._corridor_profile(nb, nb_pos)
                    corr_delta   = next_corr["value"] - cur_corr["value"]
                    score       += 2.5 * max(0.0, corr_delta)
                    if next_corr["best_run"] >= 4:
                        score += 12.0
                    if next_corr["best_run"] >= 5:
                        score += 8.0

            elif mv.move_type == enums.MoveType.SEARCH and mv.search_loc is not None:
                x, y = mv.search_loc
                p    = float(self.belief[y * BOARD + x])

                # True expected-value formula: 4P + (-2)(1-P) = 6P - 2
                ev    = 6.0 * p - 2.0
                score += 10.0 * ev

                # Situational adjustments
                if lead <= -4:
                    score += 6.0 * p    # desperate: heavily reward rat capture
                elif lead <= 1:
                    score += 2.0 * p
                elif lead >= 4:
                    score -= 2.0 * p    # comfortable lead: be conservative

            else:  # PLAIN
                score += 0.5
                if nb is not None and imm == 0:
                    next_rat_dist = self._best_belief_distance(nb.player_worker.position)
                    rat_progress  = rat_dist - next_rat_dist
                    score        -= 2.5
                    score        += 2.0 * rat_progress

                    # Moving toward positions with carpet chain potential
                    carpet_pot = self._carpet_potential_score(nb)
                    score += 0.45 * carpet_pot

                    next_corr  = self._corridor_profile(nb, nb.player_worker.position)
                    corr_delta = next_corr["value"] - cur_corr["value"]
                    score += 1.8 * max(0.0, corr_delta)

                    if early_game and not endgame:
                        setup = self._setup_plan_profile(nb)
                        score += 1.2 * setup["value"]
                        if setup["best_roll"] >= 4:
                            score += 8.0
                        if setup["best_roll"] >= 5:
                            score += 6.0

                    if rat_progress <= 0 and two_move_cap <= 2:
                        score -= 4.0

            # Novelty bonus (anti-loop)
            dest = self._move_destination(board_obj, mv)
            if dest is not None:
                score += self._novelty_bonus(dest)

            scored.append((score, mv))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    def _pick_safe_default(self, board_obj, valid_moves):
        ordered = self._order_moves(board_obj, valid_moves)
        for mv in ordered:
            if not self._carpet_is_acceptable(board_obj, mv):
                continue
            if self._valid_search(board_obj, mv, self.belief):
                return mv
        for mv in ordered:
            if self._carpet_is_acceptable(board_obj, mv):
                return mv
        return ordered[0] if ordered else random.choice(valid_moves)

    # ═══════════════════════════════════════════════════════════════════════════
    # CARPET HARD GATE  (from Agent 6, improved)
    # ═══════════════════════════════════════════════════════════════════════════

    def _carpet_is_acceptable(self, board_obj, mv) -> bool:
        """
        Hard veto on carpet moves that are net-negative or premature.

        Never roll=1 (loses a point) unless endgame + desperate.
        Avoid roll=2 (2 pts) if a roll ≥4 is reachable within 2 moves.
        Avoid any roll where extending by 1 gives ≥4 extra points (unless endgame).
        """
        if mv.move_type != enums.MoveType.CARPET:
            return True

        length  = mv.roll_length
        pts     = _cp(length)
        endgame = self._is_endgame(board_obj)

        # Rule 1: roll=1 is always wrong unless desperate endgame
        if length == 1:
            return endgame and self._is_desperate(board_obj)

        # Rule 2: don't cash a small roll if a much better one is imminent
        if not endgame:
            two_move = self._best_carpet_in_two_moves(board_obj)
            if length <= 2 and two_move >= 6:
                return False
            # Rule 3: extending by 1 would gain ≥4 extra pts — wait
            gain_from_extending = _cp(length + 1) - pts
            if gain_from_extending >= 4:
                return False

        return True

    # ═══════════════════════════════════════════════════════════════════════════
    # SEARCH VALIDITY  (continuous urgency, phase-aware)
    # ═══════════════════════════════════════════════════════════════════════════

    def _valid_search(self, board_obj, mv, belief) -> bool:
        if mv.move_type != enums.MoveType.SEARCH or mv.search_loc is None:
            return True

        x, y   = mv.search_loc
        p      = float(belief[y * BOARD + x])
        lead   = board_obj.player_worker.points - board_obj.opponent_worker.points
        my_pos = board_obj.player_worker.position
        dist   = abs(x - my_pos[0]) + abs(y - my_pos[1])
        top_p  = float(np.max(belief))
        turns  = board_obj.player_worker.turns_left

        # Continuous urgency: 1.0 = plenty of time, 0.0 = last turn
        urgency      = max(0.0, min(1.0, turns / ENDGAME_TURNS))
        base_thresh  = 0.64

        if lead <= -4:
            base_thresh = 0.44
        elif lead <= -2:
            base_thresh = 0.50
        elif lead <= 1:
            base_thresh = 0.54
        elif lead <= 3:
            base_thresh = 0.62
        elif lead >= 4:
            base_thresh = 0.69

        # Interpolate toward 0.38 in desperate endgame
        threshold = base_thresh * urgency + 0.38 * (1.0 - urgency)

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
        Block search if a genuinely profitable carpet is available.
        NEVER counts roll=1 as strong (it loses a point).
        """
        if lead < 2:
            return False

        my_pts = board_obj.player_worker.points
        best_gain = 0

        for mv in board_obj.get_valid_moves(exclude_search=True):
            if mv.move_type != enums.MoveType.CARPET or mv.roll_length < 2:
                continue
            nb = board_obj.forecast_move(mv)
            if nb is None:
                continue
            gain = nb.player_worker.points - my_pts
            if gain > best_gain:
                best_gain = gain

        if lead >= 4 and best_gain >= 4  and p < max(0.66, top_p - 0.02):
            return True
        if lead >= 2 and best_gain >= 6  and p < 0.72:
            return True
        if lead >= 3 and best_gain >= 10 and p < 0.80:
            return True

        return False

    # ═══════════════════════════════════════════════════════════════════════════
    # BELIEF STATE
    # ═══════════════════════════════════════════════════════════════════════════

    def _compute_spawn_prior(self):
        if self._spawn_prior_cache is not None:
            return self._spawn_prior_cache.copy()
        if self.T is None:
            return np.ones(64, dtype=np.float64) / 64.0
        b = np.zeros(64, dtype=np.float64)
        b[0] = 1.0
        for _ in range(1000):
            b = b @ self.T
        s = b.sum()
        result = b / s if s > 0 else np.ones(64) / 64.0
        self._spawn_prior_cache = result
        return result.copy()

    def _predict(self):
        if self.T is None:
            return
        self.belief = self.belief @ self.T
        s = self.belief.sum()
        self.belief = self.belief / s if s > 0 else np.ones(64) / 64.0

    def _update(self, board_obj, noise, est_dist):
        """
        Full Bayesian update including temporal smoothing from Agent 4.
        Combines current observation with propagated history of past observations.
        """
        obs       = np.zeros(64, dtype=np.float64)
        my_pos    = board_obj.player_worker.position
        noise_idx = int(noise)

        for i in range(64):
            x, y   = i % BOARD, i // BOARD
            actual = abs(x - my_pos[0]) + abs(y - my_pos[1])
            cell   = board_obj.get_cell((x, y))

            # Distance likelihood (fold clamped negatives into dist=0)
            p_dist = 0.0
            for off, prob in [(-1, 0.12), (0, 0.70), (1, 0.12), (2, 0.06)]:
                if max(0, actual + off) == est_dist:
                    p_dist += prob

            p_noise = NOISE_PROBS.get(cell, NOISE_PROBS[enums.Cell.SPACE])[noise_idx]
            obs[i]  = p_dist * p_noise

        # Temporal smoothing: weight obs by compatibility with recent history
        if self.T is not None and self.obs_likelihood_history:
            history_support = np.ones(64, dtype=np.float64)
            for age, prev_obs in enumerate(reversed(self.obs_likelihood_history), start=1):
                trans    = self.T if age == 1 else self.T2
                if trans is None:
                    break
                compat   = prev_obs @ trans
                exponent = 0.22 if age == 1 else 0.12
                history_support *= np.power(np.clip(compat, 1e-9, None), exponent)
            obs *= history_support

        self.belief *= obs
        s = self.belief.sum()
        self.belief = self.belief / s if s > 0 else np.ones(64) / 64.0
        self.obs_likelihood_history.append(obs.copy())

    def _handle_respawn(self, board_obj):
        ps = board_obj.player_search
        os = board_obj.opponent_search

        if ps != self.last_player_search or os != self.last_opponent_search:
            self.last_player_search   = ps
            self.last_opponent_search = os

            if ps[0] is not None:
                loc, success = ps
                if success:
                    # We caught the rat — reset to spawn prior for new rat
                    self.belief = self._compute_spawn_prior()
                    self.obs_likelihood_history.clear()
                    self.failed_search_cooldown = 0
                    self.last_failed_search_loc = None
                else:
                    # Confirmed miss — rule out that cell
                    self.last_failed_search_loc = loc
                    self.failed_search_cooldown = 4

            if os[0] is not None:
                if os[1]:
                    # Opponent caught the rat — new rat spawned
                    self.belief = self._compute_spawn_prior()
                    self.obs_likelihood_history.clear()
                else:
                    # Opponent missed — reduce that cell
                    ox, oy = os[0]
                    self.belief[oy * BOARD + ox] *= 0.01
                    s = self.belief.sum()
                    self.belief = self.belief / s if s > 0 else np.ones(64) / 64.0

    # ═══════════════════════════════════════════════════════════════════════════
    # CARPET CHAIN ANALYSIS  (from Agent 6, integrated with Agent 4 profiles)
    # ═══════════════════════════════════════════════════════════════════════════

    def _count_primed_chain(self, board_obj, pos, direction) -> int:
        """Count consecutive PRIMED cells from pos in direction (not including pos)."""
        count   = 0
        current = pos
        pl_pos  = board_obj.player_worker.position
        op_pos  = board_obj.opponent_worker.position
        for _ in range(7):
            current = enums.loc_after_direction(current, direction)
            x, y = current
            if not (0 <= x < BOARD and 0 <= y < BOARD):
                break
            if current in (pl_pos, op_pos):
                break
            if board_obj.get_cell(current) != enums.Cell.PRIMED:
                break
            count += 1
        return count

    def _carpet_potential_score(self, board_obj) -> float:
        """
        Reward positions from which a long carpet is already reachable.
        Convex: 4-chain is worth much more than 2x a 2-chain.
        """
        pos   = board_obj.player_worker.position
        score = 0.0
        for direction in enums.Direction:
            chain = self._count_primed_chain(board_obj, pos, direction)
            if chain == 0:
                continue
            pts = _cp(chain)
            if pts > 0:
                score += pts * 1.2
            elif pts < 0:
                score -= 2.0   # penalty for length-1 chain (trap)
            # Extension value: one more PRIME → meaningfully better?
            ext_pts = _cp(chain + 1)
            if ext_pts - pts >= 3:
                score += (ext_pts - pts) * 0.55
        return score

    def _best_carpet_in_two_moves(self, board_obj) -> int:
        """Max carpet gain reachable within 2 plies (PRIME+CARPET, MOVE+CARPET etc.)"""
        my_pts = board_obj.player_worker.points
        best   = 0
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
                gain = nb2.player_worker.points - my_pts
                if gain > best:
                    best = gain
        return best

    def _carpet_profile(self, board_obj) -> dict:
        """
        Unified carpet profile combining both gain-based AND chain-geometry info.
        Sort: gain DESC, roll_length DESC — so roll=1 (gain=-1) is always last.
        """
        my_pts  = board_obj.player_worker.points
        options = []
        try:
            moves = board_obj.get_valid_moves(exclude_search=True)
        except Exception:
            return {"best_gain": 0, "best_roll": 0, "line_value": 0.0,
                    "long4_count": 0, "long5_count": 0}

        for mv in moves:
            if mv.move_type != enums.MoveType.CARPET:
                continue
            nb = board_obj.forecast_move(mv)
            if nb is None:
                continue
            gain = nb.player_worker.points - my_pts
            options.append((mv.roll_length, gain))

        if not options:
            return {"best_gain": 0, "best_roll": 0, "line_value": 0.0,
                    "long4_count": 0, "long5_count": 0}

        # Gain DESC, then roll DESC — ensures roll=1 (gain=-1) is last
        options.sort(key=lambda item: (item[1], item[0]), reverse=True)
        best_roll, best_gain = options[0]
        sec_gain  = options[1][1] if len(options) > 1 else 0
        long4     = sum(1 for r, _ in options if r >= 4)
        long5     = sum(1 for r, _ in options if r >= 5)
        line_val  = best_gain + 0.55 * sec_gain + 2.5 * long4 + 3.5 * long5

        return {"best_gain": best_gain, "best_roll": best_roll,
                "line_value": line_val, "long4_count": long4, "long5_count": long5}

    # ═══════════════════════════════════════════════════════════════════════════
    # CORRIDOR + SETUP PROFILES  (from Agent 4)
    # ═══════════════════════════════════════════════════════════════════════════

    def _corridor_profile(self, board_obj, pos) -> dict:
        """Longest unblocked run in any cardinal direction — rewards open lines."""
        pl  = board_obj.player_worker.position
        opp = board_obj.opponent_worker.position
        best_run = best_primed = 0

        for direction in enums.Direction:
            loc = pos
            run = primed = 0
            for _ in range(7):
                loc  = enums.loc_after_direction(loc, direction)
                x, y = loc
                if not (0 <= x < BOARD and 0 <= y < BOARD):
                    break
                if loc in (pl, opp):
                    break
                cell = board_obj.get_cell(loc)
                if cell in (enums.Cell.BLOCKED, enums.Cell.CARPET):
                    break
                run += 1
                if cell == enums.Cell.PRIMED:
                    primed += 1
            if run > best_run or (run == best_run and primed > best_primed):
                best_run    = run
                best_primed = primed

        value = best_run + 0.8 * best_primed
        if best_run >= 4:
            value += 3.0
        if best_run >= 5:
            value += 4.0
        return {"best_run": best_run, "best_primed": best_primed, "value": value}

    def _setup_plan_profile(self, board_obj, depth=2) -> dict:
        """
        2-ply look-ahead for carpet setup value.
        Returns {"value": float, "best_roll": int}.
        """
        profile    = self._carpet_profile(board_obj)
        corridor   = self._corridor_profile(board_obj, board_obj.player_worker.position)
        base_value = 1.2 * profile["line_value"] + 1.0 * corridor["value"]
        if profile["best_roll"] >= 4:
            base_value += 8.0
        if profile["best_roll"] >= 5:
            base_value += 8.0
        if depth <= 0:
            return {"value": base_value, "best_roll": profile["best_roll"]}

        try:
            moves = board_obj.get_valid_moves(exclude_search=True)
        except Exception:
            return {"value": base_value, "best_roll": profile["best_roll"]}

        # Prioritise: PRIME > PLAIN > long CARPETs (skip short carpets)
        candidates = []
        for mv in moves:
            if mv.move_type == enums.MoveType.CARPET and mv.roll_length < 4:
                continue
            if mv.move_type not in (enums.MoveType.PLAIN,
                                    enums.MoveType.PRIME,
                                    enums.MoveType.CARPET):
                continue
            pri = {enums.MoveType.PRIME: 3, enums.MoveType.PLAIN: 2}.get(
                mv.move_type, 1 + mv.roll_length)
            candidates.append((pri, mv))

        candidates.sort(reverse=True, key=lambda item: item[0])

        best_val  = base_value
        best_roll = profile["best_roll"]
        my_pts    = board_obj.player_worker.points

        for _, mv in candidates[:6]:
            nb = board_obj.forecast_move(mv)
            if nb is None:
                continue
            child = self._setup_plan_profile(nb, depth - 1)
            gain  = nb.player_worker.points - my_pts
            value = child["value"] + 2.5 * gain

            if mv.move_type == enums.MoveType.PRIME:
                value += 4.0
            elif mv.move_type == enums.MoveType.PLAIN:
                value += 1.5
            elif mv.move_type == enums.MoveType.CARPET and mv.roll_length >= 4:
                value += 6.0 + 2.0 * mv.roll_length

            c_roll = max(child["best_roll"],
                         mv.roll_length if mv.move_type == enums.MoveType.CARPET else 0)
            if value > best_val or (abs(value - best_val) < 1e-6 and c_roll > best_roll):
                best_val  = value
                best_roll = c_roll

        return {"value": best_val, "best_roll": best_roll}

    # ═══════════════════════════════════════════════════════════════════════════
    # GAME-PHASE HELPERS
    # ═══════════════════════════════════════════════════════════════════════════

    def _is_endgame(self, board_obj) -> bool:
        return board_obj.player_worker.turns_left <= ENDGAME_TURNS

    def _is_desperate(self, board_obj) -> bool:
        lead = board_obj.player_worker.points - board_obj.opponent_worker.points
        return lead <= -4 and board_obj.player_worker.turns_left <= 20

    # ═══════════════════════════════════════════════════════════════════════════
    # SMALL HELPERS
    # ═══════════════════════════════════════════════════════════════════════════

    def _top_k_belief(self, belief, k=10):
        idx   = np.argsort(belief)[-k:]
        new_b = np.zeros_like(belief)
        new_b[idx] = belief[idx]
        s = new_b.sum()
        return new_b / s if s > 0 else belief

    def _race_value(self, my_pos, opp_pos, belief) -> float:
        idx = np.argsort(belief)[-6:]
        val = 0.0
        for i in idx:
            p    = belief[i]
            x, y = i % BOARD, i // BOARD
            val += p * ((abs(x - opp_pos[0]) + abs(y - opp_pos[1])) -
                        (abs(x - my_pos[0])  + abs(y - my_pos[1])))
        return val

    def _mobility_value(self, board_obj) -> float:
        try:
            return min(len(board_obj.get_valid_moves(exclude_search=True)), 10)
        except Exception:
            return 0.0

    def _best_belief_distance(self, pos) -> float:
        idx    = int(np.argmax(self.belief))
        target = (idx % BOARD, idx // BOARD)
        return abs(target[0] - pos[0]) + abs(target[1] - pos[1])

    def _anti_loop_eval(self, my_pos) -> float:
        if not self.position_history:
            return 0.0
        hist = list(self.position_history)
        # Penalise exact repeats
        repeats = sum(1 for p in hist if p == my_pos)
        # Also penalise A-B-A-B oscillation
        oscillate = 0
        if len(hist) >= 4:
            oscillate = sum(
                1 for i in range(len(hist) - 2) if hist[i] == hist[i + 2]
            )
        return -2.5 * max(0, repeats - 2) - 1.5 * max(0, oscillate - 2)

    def _novelty_bonus(self, pos) -> float:
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
        return nb.player_worker.position if nb is not None else None
