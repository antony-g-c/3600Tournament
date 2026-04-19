#midir
from collections import deque
from collections.abc import Callable
from typing import Optional, Tuple, Dict
import random
import time
import numpy as np

from game import board, enums


BOARD_N = 8
N_CELLS = 64
ENDGAME_TURNS = 15

# Carpet payoff from the rulebook
CARPET_PTS = {1: -1, 2: 2, 3: 4, 4: 6, 5: 10, 6: 15, 7: 21}

# Noise model
NOISE_PROBS = {
    enums.Cell.BLOCKED: (0.5, 0.3, 0.2),
    enums.Cell.SPACE:   (0.7, 0.15, 0.15),
    enums.Cell.PRIMED:  (0.1, 0.8, 0.1),
    enums.Cell.CARPET:  (0.1, 0.1, 0.8),
}


def _cp(n: int) -> int:
    return CARPET_PTS.get(min(max(1, n), 7), 21)


class PlayerAgent:
    """
    Rebalanced midir:
    - restores stronger HMM rat tracking / search behavior
    - keeps opponent adaptation
    - improves carpet conversion without over-forcing
    - protects leads with safer cash-out / denial choices
    - exits dense carpet toward the most open uncarpeted side
    - avoids excessive search suppression
    """

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        self.turn = 0

        # Belief state
        self.T = None
        self.T2 = None
        self._spawn_prior_cache = None
        if transition_matrix is not None:
            self.T = np.array(transition_matrix, dtype=np.float64)
            self.T2 = self.T @ self.T
            self.belief = self._spawn_prior()
        else:
            self.belief = np.ones(N_CELLS, dtype=np.float64) / N_CELLS

        # Search memory
        self.last_player_search = None
        self.last_opponent_search = None
        self.last_failed_search_loc = None
        self.failed_search_cooldown = 0

        # Observation memory
        self.obs_likelihood_history = deque(maxlen=3)

        # Position / loop memory
        self.position_history = deque(maxlen=12)
        self.search_history = deque(maxlen=5)
        self.prev_pos = None
        self.prev_prev_pos = None
        self.same_tile_streak = 0
        self.no_gain_streak = 0
        self.score_history = deque(maxlen=8)

        # Post-carpet / roaming memory
        self.last_move_was_carpet = False
        self.last_carpet_turn = -100
        self.last_non_carpet_turn = -100

        # Corridor tracking
        self.corridor_signature = None
        self.corridor_age = 0
        self.corridor_run = 0

        # Opponent adaptation stats
        self.opp_move_type_counts = {
            enums.MoveType.PLAIN: 0,
            enums.MoveType.PRIME: 0,
            enums.MoveType.CARPET: 0,
            enums.MoveType.SEARCH: 0,
        }
        self.opp_carpet_rolls = deque(maxlen=12)
        self.opp_search_successes = 0
        self.opp_search_attempts = 0
        self.opp_search_probs = deque(maxlen=10)
        self.opp_move_toward_rat = deque(maxlen=10)
        self.opp_disruption_scores = deque(maxlen=10)

        # Previous observed opponent state for rough inference
        self.prev_opp_pos = None
        self.prev_opp_points = None
        self.prev_carpet_count = None
        self.prev_primed_count = None

    def commentate(self):
        return ""

    # =========================================================
    # MAIN ENTRY
    # =========================================================

    def play(self, board: board.Board, sensor_data: Tuple, time_left: Callable):
        self.turn += 1
        start = time.time()

        my_pos = board.player_worker.position
        my_points = board.player_worker.points

        self.prev_prev_pos = self.prev_pos
        if self.prev_pos == my_pos:
            self.same_tile_streak += 1
        else:
            self.same_tile_streak = 0
        self.prev_pos = my_pos
        self.position_history.append(my_pos)

        self.score_history.append(my_points)
        if len(self.score_history) >= 2 and self.score_history[-1] <= self.score_history[-2]:
            self.no_gain_streak += 1
        else:
            self.no_gain_streak = 0

        if self.failed_search_cooldown > 0:
            self.failed_search_cooldown -= 1

        self._handle_respawn_and_opponent(board)

        if self.T is not None and sensor_data is not None and len(sensor_data) == 2:
            noise, dist = sensor_data
            self._predict()
            self._update(board, noise, dist)

        self._update_corridor_tracking(board)

        valid_moves = board.get_valid_moves(exclude_search=False)
        if not valid_moves:
            fallback = board.get_valid_moves()
            return random.choice(fallback) if fallback else None

        filtered = self._filter_moves(board, valid_moves)
        if not filtered:
            filtered = [m for m in valid_moves if m.move_type != enums.MoveType.SEARCH]
            if not filtered:
                filtered = valid_moves

        forced_carpet = self._should_force_carpet_now(board, filtered if filtered else valid_moves)
        if forced_carpet is not None:
            self._record_move_bookkeeping(forced_carpet)
            self._store_opp_snapshot(board)
            return forced_carpet

        ordered = self._order_moves(board, filtered)
        if not ordered:
            mv = random.choice(filtered)
            self._record_move_bookkeeping(mv)
            self._store_opp_snapshot(board)
            return mv

        best_move = self._pick_safe_default(board, ordered)

        depth_plan = [1, 2, 3]
        if self._should_search_deeper(board):
            depth_plan = [1, 2, 3, 4]

        for depth in depth_plan:
            if time.time() - start > 0.090:
                break
            mv = self._search_root(board, depth, start)
            if mv is not None:
                best_move = mv

        # Final mild cash-out guard
        forced_end = self._should_force_carpet_now(board, ordered)
        if forced_end is not None:
            best_move = forced_end

        self._record_move_bookkeeping(best_move)
        self._store_opp_snapshot(board)
        return best_move

    def _record_move_bookkeeping(self, mv):
        self.last_move_was_carpet = (mv.move_type == enums.MoveType.CARPET)
        if self.last_move_was_carpet:
            self.last_carpet_turn = self.turn
        else:
            self.last_non_carpet_turn = self.turn
        self.search_history.append(1 if mv.move_type == enums.MoveType.SEARCH else 0)

    # =========================================================
    # HMM BELIEF
    # =========================================================

    def _spawn_prior(self):
        if self._spawn_prior_cache is not None:
            return self._spawn_prior_cache.copy()

        if self.T is None:
            return np.ones(N_CELLS, dtype=np.float64) / N_CELLS

        prior = np.zeros(N_CELLS, dtype=np.float64)
        prior[0] = 1.0
        for _ in range(1000):
            prior = prior @ self.T
        s = prior.sum()
        result = prior / s if s > 0 else np.ones(N_CELLS, dtype=np.float64) / N_CELLS
        self._spawn_prior_cache = result.copy()
        return result

    def _predict(self):
        if self.T is None:
            return
        self.belief = self.belief @ self.T
        self._renorm()

    def _update(self, board_obj, noise, est_dist):
        """
        Stronger rat model:
        - transition prediction
        - distance + noise likelihood
        - temporal smoothing using recent observation likelihoods
        """
        my_pos = board_obj.player_worker.position
        noise_idx = self._noise_index(noise)
        obs = np.zeros(N_CELLS, dtype=np.float64)

        for idx in range(N_CELLS):
            x, y = idx % BOARD_N, idx // BOARD_N
            cell_type = board_obj.get_cell((x, y))
            true_dist = abs(x - my_pos[0]) + abs(y - my_pos[1])

            p_noise = NOISE_PROBS.get(cell_type, NOISE_PROBS[enums.Cell.SPACE])[noise_idx]
            p_dist = self._distance_prob(true_dist, int(est_dist))
            obs[idx] = p_noise * p_dist

        if self.T is not None and self.obs_likelihood_history:
            hist_support = np.ones(N_CELLS, dtype=np.float64)
            for age, prev_obs in enumerate(reversed(self.obs_likelihood_history), start=1):
                trans = self.T if age == 1 else self.T2
                if trans is None:
                    break
                compat = prev_obs @ trans
                exponent = 0.22 if age == 1 else 0.12
                hist_support *= np.power(np.clip(compat, 1e-9, None), exponent)
            obs *= hist_support

        self.belief *= obs
        s = self.belief.sum()
        if s > 0:
            self.belief /= s
        else:
            self.belief = self._spawn_prior()

        self.obs_likelihood_history.append(obs.copy())

    def _noise_index(self, noise) -> int:
        if hasattr(noise, "value"):
            try:
                return int(noise.value)
            except Exception:
                pass
        if hasattr(noise, "name"):
            name = str(noise.name).lower()
        else:
            name = str(noise).lower()

        if "scratch" in name:
            return 1
        if "squeal" in name:
            return 2
        return 0

    def _distance_prob(self, true_dist: int, reported_dist: int) -> float:
        p = 0.0
        for off, prob in [(-1, 0.12), (0, 0.70), (1, 0.12), (2, 0.06)]:
            if max(0, true_dist + off) == reported_dist:
                p += prob
        return max(p, 1e-9)

    def _renorm(self):
        s = self.belief.sum()
        if s > 0:
            self.belief /= s
        else:
            self.belief = np.ones(N_CELLS, dtype=np.float64) / N_CELLS

    # =========================================================
    # RESPAWN + OPPONENT ADAPTATION
    # =========================================================

    def _handle_respawn_and_opponent(self, board_obj):
        player_search = board_obj.player_search
        opponent_search = board_obj.opponent_search

        if player_search != self.last_player_search or opponent_search != self.last_opponent_search:
            self.last_player_search = player_search
            self.last_opponent_search = opponent_search

            if player_search[0] is not None:
                loc, success = player_search
                if success:
                    self.belief = self._spawn_prior()
                    self.obs_likelihood_history.clear()
                    self.last_failed_search_loc = None
                    self.failed_search_cooldown = 0
                else:
                    self.last_failed_search_loc = loc
                    self.failed_search_cooldown = 4
                    x, y = loc
                    self.belief[y * BOARD_N + x] *= 0.01
                    self._renorm()

            if opponent_search[0] is not None:
                loc, success = opponent_search
                self.opp_move_type_counts[enums.MoveType.SEARCH] += 1
                self.opp_search_attempts += 1
                if loc is not None:
                    x, y = loc
                    p = float(self.belief[y * BOARD_N + x])
                    self.opp_search_probs.append(p)

                if success:
                    self.opp_search_successes += 1
                    self.belief = self._spawn_prior()
                    self.obs_likelihood_history.clear()
                else:
                    x, y = loc
                    self.belief[y * BOARD_N + x] *= 0.01
                    self._renorm()

        self._infer_opponent_style(board_obj)

    def _infer_opponent_style(self, board_obj):
        try:
            opp_pos = board_obj.opponent_worker.position
            my_pos = board_obj.player_worker.position
            opp_points = board_obj.opponent_worker.points
        except Exception:
            return

        best_loc, _ = self._best_search_cell()
        d_opp = abs(best_loc[0] - opp_pos[0]) + abs(best_loc[1] - opp_pos[1])
        d_me = abs(best_loc[0] - my_pos[0]) + abs(best_loc[1] - my_pos[1])
        self.opp_move_toward_rat.append(max(-2.0, min(2.0, d_me - d_opp)))

        my_profile = self._carpet_profile(board_obj)
        disruption = 0.0
        if my_profile["best_roll"] >= 4:
            disruption += 1.0
        if abs(my_pos[0] - opp_pos[0]) + abs(my_pos[1] - opp_pos[1]) <= 2:
            disruption += 0.8
        self.opp_disruption_scores.append(disruption)

        carpet_count = self._count_cells(board_obj, enums.Cell.CARPET)
        primed_count = self._count_cells(board_obj, enums.Cell.PRIMED)

        if self.prev_opp_pos is not None and self.prev_opp_points is not None:
            point_delta = opp_points - self.prev_opp_points
            moved = opp_pos != self.prev_opp_pos

            if self.last_opponent_search is not None and self.last_opponent_search[0] is not None:
                pass
            else:
                if point_delta >= 2:
                    self.opp_move_type_counts[enums.MoveType.CARPET] += 1
                    if point_delta in CARPET_PTS.values():
                        for k, v in CARPET_PTS.items():
                            if v == point_delta:
                                self.opp_carpet_rolls.append(k)
                                break
                elif point_delta == 1:
                    self.opp_move_type_counts[enums.MoveType.PRIME] += 1
                elif moved:
                    self.opp_move_type_counts[enums.MoveType.PLAIN] += 1

            if (
                self.prev_carpet_count is not None
                and self.prev_primed_count is not None
                and carpet_count > self.prev_carpet_count
            ):
                self.opp_move_type_counts[enums.MoveType.CARPET] += 1

    def _store_opp_snapshot(self, board_obj):
        try:
            self.prev_opp_pos = board_obj.opponent_worker.position
            self.prev_opp_points = board_obj.opponent_worker.points
            self.prev_carpet_count = self._count_cells(board_obj, enums.Cell.CARPET)
            self.prev_primed_count = self._count_cells(board_obj, enums.Cell.PRIMED)
        except Exception:
            pass

    def _opponent_profile(self) -> Dict[str, float]:
        total_known = sum(self.opp_move_type_counts.values())
        if total_known <= 0:
            total_known = 1

        carpet_rate = self.opp_move_type_counts[enums.MoveType.CARPET] / total_known
        prime_rate = self.opp_move_type_counts[enums.MoveType.PRIME] / total_known
        search_rate = self.opp_move_type_counts[enums.MoveType.SEARCH] / total_known

        avg_roll = float(sum(self.opp_carpet_rolls) / len(self.opp_carpet_rolls)) if self.opp_carpet_rolls else 0.0
        avg_search_p = float(sum(self.opp_search_probs) / len(self.opp_search_probs)) if self.opp_search_probs else 0.0
        rat_chase = float(sum(self.opp_move_toward_rat) / len(self.opp_move_toward_rat)) if self.opp_move_toward_rat else 0.0
        disruption = float(sum(self.opp_disruption_scores) / len(self.opp_disruption_scores)) if self.opp_disruption_scores else 0.0

        return {
            "carpet_rate": carpet_rate,
            "prime_rate": prime_rate,
            "search_rate": search_rate,
            "avg_roll": avg_roll,
            "avg_search_p": avg_search_p,
            "rat_chase": rat_chase,
            "disruption": disruption,
        }

    def _is_bot3_like_opponent(self) -> bool:
        total_known = sum(self.opp_move_type_counts.values())
        if total_known < 8:
            return False

        profile = self._opponent_profile()
        carpets_present = profile["carpet_rate"] >= 0.14 or profile["avg_roll"] >= 2.4
        roll_ok = profile["avg_roll"] == 0.0 or profile["avg_roll"] <= 4.4
        return (
            profile["search_rate"] <= 0.14
            and profile["prime_rate"] >= 0.34
            and carpets_present
            and roll_ok
        )

    def _is_board_pressure_opponent(self) -> bool:
        total_known = sum(self.opp_move_type_counts.values())
        if total_known < 8:
            return False

        profile = self._opponent_profile()
        return (
            profile["search_rate"] <= 0.20
            and profile["prime_rate"] >= 0.20
            and (profile["carpet_rate"] >= 0.14 or profile["avg_roll"] >= 2.2)
        )

    def _board_pressure_from_profile(self, profile: Dict[str, float], future_gain: float = 0.0) -> float:
        pressure = 0.55 * profile["line_value"] + 0.25 * max(0.0, future_gain)
        if profile["best_roll"] >= 4:
            pressure += 8.0
        if profile["best_roll"] >= 5:
            pressure += 10.0
        if profile["best_roll"] >= 6:
            pressure += 12.0
        if future_gain >= 6:
            pressure += 4.0
        if future_gain >= 10:
            pressure += 6.0
        return pressure

    # =========================================================
    # CORRIDOR TRACKING / CONVERSION URGENCY
    # =========================================================

    def _count_primed_run_from(self, board_obj, start_pos, direction) -> int:
        run = 0
        cur = start_pos
        my_pos = board_obj.player_worker.position
        opp_pos = board_obj.opponent_worker.position

        for _ in range(7):
            cur = enums.loc_after_direction(cur, direction)
            x, y = cur
            if not (0 <= x < BOARD_N and 0 <= y < BOARD_N):
                break
            if cur == my_pos or cur == opp_pos:
                break
            cell = board_obj.get_cell(cur)
            if cell != enums.Cell.PRIMED:
                break
            run += 1

        return run

    def _best_corridor_info(self, board_obj):
        pos = board_obj.player_worker.position
        best = {
            "run": 0,
            "dir": None,
            "endpoint": pos,
            "signature": None,
            "cells": [pos],
        }

        for direction in enums.Direction:
            run = self._count_primed_run_from(board_obj, pos, direction)
            if run > best["run"]:
                cur = pos
                cells = [pos]
                for _ in range(run):
                    cur = enums.loc_after_direction(cur, direction)
                    cells.append(cur)

                best = {
                    "run": run,
                    "dir": direction,
                    "endpoint": cur,
                    "signature": (pos, direction, run),
                    "cells": cells,
                }

        return best

    def _update_corridor_tracking(self, board_obj):
        info = self._best_corridor_info(board_obj)
        sig = info["signature"]
        self.corridor_run = info["run"]

        if info["run"] < 3 or sig is None:
            self.corridor_signature = None
            self.corridor_age = 0
            return

        if sig == self.corridor_signature:
            self.corridor_age += 1
        else:
            self.corridor_signature = sig
            self.corridor_age = 1

    def _opponent_corridor_pressure(self, board_obj) -> int:
        info = self._best_corridor_info(board_obj)
        if info["run"] < 3:
            return 99

        opp = board_obj.opponent_worker.position
        best_dist = 99
        for cell in info["cells"]:
            d = abs(cell[0] - opp[0]) + abs(cell[1] - opp[1])
            if d < best_dist:
                best_dist = d
        return best_dist

    def _best_carpet_move(self, board_obj, moves):
        best_mv = None
        best_roll_pts = -10
        best_gain = -10**9
        my_pts = board_obj.player_worker.points

        for mv in moves:
            if mv.move_type != enums.MoveType.CARPET:
                continue
            try:
                nb = board_obj.forecast_move(mv)
                if nb is None:
                    continue
                gain = nb.player_worker.points - my_pts
                roll_pts = _cp(getattr(mv, "roll_length", 0))
                if roll_pts > best_roll_pts or (roll_pts == best_roll_pts and gain > best_gain):
                    best_roll_pts = roll_pts
                    best_gain = gain
                    best_mv = mv
            except Exception:
                continue

        return best_mv, best_roll_pts, best_gain

    def _best_future_carpet_score(self, board_obj) -> int:
        best = -10
        try:
            moves = board_obj.get_valid_moves(exclude_search=True)
        except Exception:
            return best

        for mv in moves:
            if mv.move_type == enums.MoveType.CARPET:
                best = max(best, _cp(getattr(mv, "roll_length", 0)))
                continue

            try:
                nb = board_obj.forecast_move(mv)
                if nb is None:
                    continue
                next_moves = nb.get_valid_moves(exclude_search=True)
            except Exception:
                continue

            for mv2 in next_moves:
                if mv2.move_type != enums.MoveType.CARPET:
                    continue
                best = max(best, _cp(getattr(mv2, "roll_length", 0)))

        return best

    def _corridor_cashout_pressure(self, board_obj) -> float:
        pressure = 0.0
        best_roll = self._carpet_profile(board_obj)["best_roll"]
        opp_dist = self._opponent_corridor_pressure(board_obj)

        if self.corridor_run >= 3:
            pressure += 6.0
            pressure += 4.0 * max(0, self.corridor_age - 1)
        if self.corridor_run >= 4:
            pressure += 8.0
        if self.corridor_run >= 5:
            pressure += 10.0
        if best_roll >= 2:
            pressure += 3.0
        if best_roll >= 4:
            pressure += 8.0
        if best_roll >= 6:
            pressure += 12.0
        if opp_dist <= 4:
            pressure += 8.0
        if opp_dist <= 3:
            pressure += 12.0
        if opp_dist <= 2:
            pressure += 18.0
        return pressure

    def _should_force_carpet_now(self, board_obj, moves):
        best_carpet, best_roll_pts, _ = self._best_carpet_move(board_obj, moves)
        if best_carpet is None:
            return None

        lead = board_obj.player_worker.points - board_obj.opponent_worker.points
        turns = board_obj.player_worker.turns_left
        peak = float(np.max(self.belief))
        future_best = self._best_future_carpet_score(board_obj)
        opp_dist = self._opponent_corridor_pressure(board_obj)
        cashout_pressure = self._corridor_cashout_pressure(board_obj)
        anti_bot3 = self._is_bot3_like_opponent()
        board_pressure_opp = self._is_board_pressure_opponent()
        my_profile = self._carpet_profile(board_obj) if anti_bot3 else None
        opp_profile_board = self._opponent_carpet_profile(board_obj)
        opp_two = self._best_opponent_carpet_in_two_moves(board_obj)
        opp_board_pressure = self._board_pressure_from_profile(opp_profile_board, opp_two)

        if best_roll_pts >= 4 and opp_board_pressure >= 16.0:
            return best_carpet
        if board_pressure_opp and best_roll_pts >= 2 and opp_board_pressure >= 22.0:
            if lead >= -1 or turns <= 14:
                return best_carpet

        if anti_bot3 and opp_profile_board is not None and my_profile is not None:
            if best_roll_pts >= 4 and (
                opp_profile_board["line_value"] >= 8.0
                or opp_profile_board["best_roll"] >= 4
                or lead >= 0
            ):
                return best_carpet
            if best_roll_pts >= 2 and lead >= 0 and turns <= 16:
                if (
                    opp_profile_board["line_value"] >= my_profile["line_value"] - 1.0
                    or opp_profile_board["best_roll"] >= 4
                ):
                    return best_carpet

        # Rule: carpet immediately if current roll is worth >= 6,
        # unless waiting one turn is overwhelmingly better.
        if best_roll_pts >= 6:
            if future_best <= best_roll_pts + 5 or opp_dist <= 3 or self.corridor_age >= 2:
                return best_carpet

        if lead >= 4 and best_roll_pts >= 4 and peak < 0.74:
            if future_best <= best_roll_pts + 4 or opp_dist <= 4 or turns <= 16:
                return best_carpet

        if lead >= 8 and turns <= 12 and best_roll_pts >= 2 and peak < 0.70:
            return best_carpet

        # Mature corridor: start cashing earlier, but not on every weak setup.
        if self.corridor_run >= 3:
            if self.corridor_age >= 3 and best_roll_pts >= 4:
                return best_carpet
            if self.corridor_age >= 4 and best_roll_pts >= 2 and opp_dist <= 3:
                return best_carpet

        # Threatened corridor: stop optimizing and cash it.
        if opp_dist <= 2 and best_roll_pts >= 2:
            return best_carpet
        if opp_dist <= 3 and best_roll_pts >= 4:
            return best_carpet

        if cashout_pressure >= 24.0 and best_roll_pts >= 4:
            return best_carpet

        return None

    # =========================================================
    # SEARCH / EXPECTIMINIMAX
    # =========================================================

    def _should_search_deeper(self, board_obj) -> bool:
        lead = board_obj.player_worker.points - board_obj.opponent_worker.points
        turns = board_obj.player_worker.turns_left
        peak = float(np.max(self.belief))
        return turns <= 16 or abs(lead) <= 4 or peak >= 0.55 or self._volatile_position(board_obj)

    def _volatile_position(self, board_obj) -> bool:
        profile = self._carpet_profile(board_obj)
        opp_threat = self._opponent_carpet_profile(board_obj)
        return (
            profile["best_roll"] >= 4
            or opp_threat["best_roll"] >= 4
            or abs(profile["line_value"] - opp_threat["line_value"]) >= 6
        )

    def _search_root(self, board_obj, depth: int, start_time: float):
        alpha = -1e18
        beta = 1e18

        moves = self._order_moves(board_obj, board_obj.get_valid_moves(exclude_search=False))
        moves = [m for m in moves if self._move_allowed(board_obj, m)]
        moves = moves[: self._root_cap_for(board_obj, depth)]

        best_move = None
        best_val = -1e18

        for mv in moves:
            if time.time() - start_time > 0.095:
                break

            nb = board_obj.forecast_move(mv)
            if nb is None:
                continue
            nb.reverse_perspective()

            val = self._tree_value(
                nb,
                self.belief.copy(),
                depth - 1,
                node="MIN",
                alpha=alpha,
                beta=beta,
                start_time=start_time,
            )

            if val > best_val:
                best_val = val
                best_move = mv
            alpha = max(alpha, best_val)

        return best_move

    def _tree_value(
        self,
        board_obj,
        belief: np.ndarray,
        depth: int,
        node: str,
        alpha: float,
        beta: float,
        start_time: float,
    ) -> float:
        if time.time() - start_time > 0.097 or depth <= 0:
            return self._evaluate(board_obj, belief)

        belief = self._top_k_belief(belief, k=10)

        if node == "MAX":
            moves = self._order_moves(board_obj, board_obj.get_valid_moves(exclude_search=False))
            moves = [m for m in moves if self._move_allowed(board_obj, m, belief)]
            moves = moves[: self._tree_cap_for(board_obj, depth)]

            best = -1e18
            for mv in moves:
                nb = board_obj.forecast_move(mv)
                if nb is None:
                    continue
                nb.reverse_perspective()
                val = self._tree_value(nb, belief, depth - 1, "CHANCE", alpha, beta, start_time)
                best = max(best, val)
                alpha = max(alpha, best)
                if alpha >= beta:
                    break
            return best if best > -1e17 else self._evaluate(board_obj, belief)

        if node == "MIN":
            moves = self._order_moves(board_obj, board_obj.get_valid_moves(exclude_search=False))
            moves = moves[: self._reply_cap_for(board_obj, depth)]

            worst = 1e18
            for mv in moves:
                nb = board_obj.forecast_move(mv)
                if nb is None:
                    continue
                nb.reverse_perspective()
                val = self._tree_value(nb, belief, depth - 1, "CHANCE", alpha, beta, start_time)
                worst = min(worst, val)
                beta = min(beta, worst)
                if alpha >= beta:
                    break
            return worst if worst < 1e17 else self._evaluate(board_obj, belief)

        if self.T is None:
            return self._tree_value(board_obj, belief, depth - 1, "MAX", alpha, beta, start_time)

        new_belief = belief @ self.T
        s = new_belief.sum()
        if s > 0:
            new_belief /= s
        else:
            new_belief = np.ones(N_CELLS, dtype=np.float64) / N_CELLS
        return self._tree_value(board_obj, new_belief, depth - 1, "MAX", alpha, beta, start_time)

    def _root_cap_for(self, board_obj, depth: int) -> int:
        if depth <= 1:
            return 12
        if self._is_endgame(board_obj):
            return 10
        return 9

    def _tree_cap_for(self, board_obj, depth: int) -> int:
        if depth >= 3:
            return 6
        return 7

    def _reply_cap_for(self, board_obj, depth: int) -> int:
        if depth >= 3:
            return 5
        return 6

    # =========================================================
    # EVALUATION
    # =========================================================

    def _evaluate(self, board_obj, belief: np.ndarray) -> float:
        my = board_obj.player_worker.points
        opp = board_obj.opponent_worker.points
        my_pos = board_obj.player_worker.position
        opp_pos = board_obj.opponent_worker.position
        turns = board_obj.player_worker.turns_left
        lead = my - opp
        endgame = self._is_endgame(board_obj)
        opp_profile = self._opponent_profile()
        anti_bot3 = self._is_bot3_like_opponent()
        board_pressure_opp = self._is_board_pressure_opponent()

        score = 0.0

        phase_mult = 30.0 + 24.0 * max(0.0, (ENDGAME_TURNS - turns) / ENDGAME_TURNS)
        score += phase_mult * lead

        expected_dist = 0.0
        for idx in range(N_CELLS):
            p = float(belief[idx])
            if p <= 0:
                continue
            x, y = idx % BOARD_N, idx // BOARD_N
            expected_dist += p * (abs(x - my_pos[0]) + abs(y - my_pos[1]))
        score += 8.0 / (1.0 + expected_dist)

        peak = float(np.max(belief))
        peak_gap = self._belief_peak_gap(belief)
        entropy = self._belief_entropy(belief)
        score += 18.0 * peak
        score += 7.0 * peak_gap
        score -= 1.7 * entropy

        score += 2.0 * self._race_value_from_belief(my_pos, opp_pos, belief)
        score += 1.8 * self._rat_access_score(my_pos, belief)
        score += 2.4 * self._mobility_value(board_obj)
        score += 1.0 * self._local_control(board_obj)

        # Light anti-looping on owned carpet only
        score += 1.8 * self._fresh_space_value(board_obj)
        score -= 2.4 * self._carpet_roam_penalty(board_obj)
        if self.turn - self.last_carpet_turn <= 3:
            score += 2.6 * self._post_carpet_exit_value(board_obj)

        my_profile = self._carpet_profile(board_obj)
        opp_profile_board = self._opponent_carpet_profile(board_obj)

        score += (4.8 if not endgame else 1.8) * my_profile["line_value"]
        score -= (4.2 if not endgame else 2.3) * opp_profile_board["line_value"]

        my_two = self._best_carpet_in_two_moves(board_obj)
        opp_two = self._best_opponent_carpet_in_two_moves(board_obj)
        score += 2.5 * my_two
        score -= 2.2 * opp_two
        my_pressure = self._board_pressure_from_profile(my_profile, my_two)
        opp_pressure = self._board_pressure_from_profile(opp_profile_board, opp_two)

        my_corr = self._corridor_profile(board_obj, my_pos)
        opp_corr = self._corridor_profile_for_opponent(board_obj)
        score += 2.0 * my_corr["value"]
        score -= 1.6 * opp_corr["value"]

        dense_pressure = self._dense_carpet_pressure_at(board_obj, my_pos)
        if dense_pressure >= 2.8:
            _, best_side_val, _ = self._best_uncarpeted_side(board_obj, my_pos)
            score -= 1.5 * dense_pressure
            score += 0.18 * max(0.0, best_side_val)

        score += 0.8 * self._corridor_cashout_pressure(board_obj)
        score += 3.0 * self._anti_loop_eval(my_pos)
        score += 0.22 * my_pressure
        score -= 0.40 * opp_pressure

        if self._is_desperate(board_obj):
            score += 8.0 * peak
            score += 4.0 / (1.0 + expected_dist)

        if lead >= 4:
            lead_scale = min(1.6, 0.18 * lead)
            score += (1.6 + 0.8 * lead_scale) * my_profile["best_gain"]
            score -= (1.0 + 0.7 * lead_scale) * opp_profile_board["line_value"]
            score -= (1.1 + 0.4 * lead_scale) * opp_two
            score += (1.0 + 0.3 * lead_scale) * max(0.0, self._rat_zone_control(board_obj, belief))

        if opp_profile["carpet_rate"] >= 0.35 or opp_profile["avg_roll"] >= 3.5:
            score -= 1.6 * opp_profile_board["line_value"]
            score -= 0.8 * opp_two

        if opp_profile["search_rate"] >= 0.18 and opp_profile["avg_search_p"] >= 0.62:
            score += 1.8 * self._rat_zone_control(board_obj, belief)

        if opp_profile["disruption"] >= 0.9:
            score += 1.3 * my_corr["value"]
            score += 0.8 * my_profile["long4_count"]

        if board_pressure_opp:
            score -= 0.45 * opp_pressure
            if my_profile["best_roll"] >= 4:
                score += 4.0

        if anti_bot3:
            score -= 2.0 * opp_profile_board["line_value"]
            score -= 1.6 * opp_two
            score += 1.2 * my_profile["best_gain"]
            if opp_profile_board["best_roll"] >= 4:
                score -= 8.0

        return score

    # =========================================================
    # MOVE ORDERING / FILTERING
    # =========================================================

    def _filter_moves(self, board_obj, moves):
        filtered = []
        for mv in moves:
            if self._move_allowed(board_obj, mv, self.belief):
                filtered.append(mv)
        return filtered

    def _move_allowed(self, board_obj, mv, belief: Optional[np.ndarray] = None) -> bool:
        if belief is None:
            belief = self.belief

        if mv.move_type == enums.MoveType.CARPET:
            return self._carpet_is_acceptable(board_obj, mv)

        if mv.move_type == enums.MoveType.SEARCH:
            return self._valid_search(board_obj, mv, belief)

        return True

    def _order_moves(self, board_obj, moves):
        scored = []
        my_pts = board_obj.player_worker.points
        lead = my_pts - board_obj.opponent_worker.points
        turns = board_obj.player_worker.turns_left
        endgame = self._is_endgame(board_obj)
        opp_profile = self._opponent_profile()
        anti_bot3 = self._is_bot3_like_opponent()
        board_pressure_opp = self._is_board_pressure_opponent()

        cur_profile = self._carpet_profile(board_obj)
        cur_opp_profile = self._opponent_carpet_profile(board_obj)
        my_pos = board_obj.player_worker.position
        best_loc, best_p = self._best_search_cell()
        peak = float(np.max(self.belief))
        peak_gap = self._belief_peak_gap(self.belief)
        entropy = self._belief_entropy(self.belief)
        rat_access_now = self._rat_access_score(my_pos, self.belief)
        opp_corridor_pressure = self._opponent_corridor_pressure(board_obj)
        cashout_pressure = self._corridor_cashout_pressure(board_obj)
        _, best_side_val, side_values = self._best_uncarpeted_side(board_obj, my_pos)
        dense_pressure_now = self._dense_carpet_pressure_at(board_obj, my_pos)
        opp_board_pressure = self._board_pressure_from_profile(cur_opp_profile, cur_opp_profile["best_gain"])
        anti_bot3_pressure = 0.0
        if anti_bot3:
            anti_bot3_pressure = 0.55 * cur_opp_profile["line_value"]
            if cur_opp_profile["best_roll"] >= 4:
                anti_bot3_pressure += 8.0
            if lead <= 2:
                anti_bot3_pressure += 4.0

        for mv in moves:
            nb = board_obj.forecast_move(mv)
            if nb is None:
                continue

            score = 0.0
            imm = nb.player_worker.points - my_pts
            score += 7.5 * imm

            if mv.move_type == enums.MoveType.CARPET:
                length = mv.roll_length
                pts = _cp(length)

                if pts < 0:
                    score -= 60.0
                else:
                    next_profile = self._carpet_profile(nb)
                    next_opp_profile = self._opponent_carpet_profile(nb)
                    denial = max(0.0, cur_opp_profile["line_value"] - next_opp_profile["line_value"])
                    self_gain = max(0.0, next_profile["line_value"] - cur_profile["line_value"])

                    score += 10.0 + 2.0 * pts
                    score += 1.8 * self_gain
                    score += 2.2 * denial
                    score += 0.45 * cashout_pressure

                    if endgame:
                        urgency = max(0.0, 1.0 - turns / ENDGAME_TURNS)
                        score += pts * 1.8 * urgency

                    if pts >= 4:
                        score += 5.0
                    if pts >= 6:
                        score += 12.0
                    if pts >= 10:
                        score += 14.0

                    if opp_profile["carpet_rate"] >= 0.35:
                        score += 1.2 * denial

                    if opp_corridor_pressure <= 3:
                        score += 8.0
                    if opp_corridor_pressure <= 2:
                        score += 10.0

                    if self._is_roaming_own_carpet(board_obj, board_obj.player_worker.position):
                        score += 5.0

                    if lead >= 4:
                        score += 1.5 * pts
                        score += 1.1 * max(0.0, cur_opp_profile["line_value"] - next_opp_profile["line_value"])
                    if lead >= 8 and pts >= 4:
                        score += 8.0
                    if dense_pressure_now >= 3.2 and pts < 6 and mv.direction != max(side_values, key=side_values.get):
                        score -= 4.0
                    score += 0.16 * opp_board_pressure
                    if denial >= 1.0:
                        score += 0.8 * denial
                    if board_pressure_opp and opp_board_pressure >= 14.0 and pts >= 2:
                        score += 4.0
                    if anti_bot3:
                        score += 1.4 * denial
                        score += 0.45 * anti_bot3_pressure
                        if pts >= 2 and anti_bot3_pressure >= 10.0:
                            score += 4.0
                        if pts >= 4 and cur_opp_profile["best_roll"] >= 4:
                            score += 6.0

            elif mv.move_type == enums.MoveType.PRIME:
                score += 3.0

                nb_pos = nb.player_worker.position
                rat_access_delta = self._rat_access_score(nb_pos, self.belief) - rat_access_now
                setup = self._setup_plan_profile(nb)
                next_profile = self._carpet_profile(nb)
                next_opp_profile = self._opponent_carpet_profile(nb)
                escape_bonus = self._dense_carpet_exit_score(board_obj, nb, my_pos, nb_pos, side_values)

                score += 1.3 * setup["value"]
                score += 1.0 * next_profile["line_value"]
                score += 1.6 * max(0.0, cur_opp_profile["line_value"] - next_opp_profile["line_value"])
                score += 4.2 * rat_access_delta
                score += 1.7 * escape_bonus

                if setup["best_roll"] >= 4:
                    score += 12.0
                if setup["best_roll"] >= 5:
                    score += 10.0

                next_corr = self._corridor_profile(nb, nb_pos)
                score += 2.0 * next_corr["value"]

                # Anti-overprime only when a real cash-out already exists
                if cur_profile["best_roll"] >= 4:
                    score -= 8.0
                if cur_profile["best_roll"] >= 6:
                    score -= 16.0
                if self.corridor_age >= 3 and cur_profile["best_roll"] >= 4:
                    score -= 8.0
                if opp_corridor_pressure <= 3 and cur_profile["best_roll"] >= 4:
                    score -= 12.0
                if opp_corridor_pressure <= 2 and cur_profile["best_roll"] >= 2:
                    score -= 10.0
                if lead >= 4 and setup["best_roll"] < 4 and rat_access_delta <= 0.12:
                    score -= 8.0
                if lead >= 8 and next_profile["best_roll"] < cur_profile["best_roll"] and next_opp_profile["line_value"] >= cur_opp_profile["line_value"]:
                    score -= 8.0
                if board_pressure_opp and opp_board_pressure >= 14.0:
                    if next_opp_profile["line_value"] >= cur_opp_profile["line_value"] - 0.75:
                        score -= 8.0
                    if setup["best_roll"] < 4:
                        score -= 4.0
                if anti_bot3 and anti_bot3_pressure >= 8.0:
                    if setup["best_roll"] < 4:
                        score -= 8.0
                    if next_opp_profile["line_value"] >= cur_opp_profile["line_value"] - 0.5:
                        score -= 6.0

            elif mv.move_type == enums.MoveType.SEARCH and mv.search_loc is not None:
                x, y = mv.search_loc
                p = float(self.belief[y * BOARD_N + x])
                ev = 6.0 * p - 2.0
                score += 11.0 * ev

                if lead <= -4:
                    score += 8.0 * p
                elif lead <= 1:
                    score += 3.0 * p
                elif lead >= 4:
                    score -= 3.0 * p

                if turns <= 8:
                    score += 5.0 * p

                # Light penalty when a very good carpet is live
                if cur_profile["best_roll"] >= 6:
                    score -= 8.0
                elif cashout_pressure >= 20.0:
                    score -= 4.0

                score += 10.0 * peak_gap
                if entropy <= 1.45:
                    score += 4.0
                if peak_gap < 0.10 and entropy > 1.70:
                    score -= 6.0
                if self.failed_search_cooldown > 0 and p < 0.72:
                    score -= 5.0
                if lead >= 6:
                    score -= 6.0 * p
                if lead >= 4 and p < 0.74:
                    score -= 6.0
                if lead >= 8 and turns > 10:
                    score -= 8.0
                if peak >= 0.78 and peak_gap >= 0.16:
                    score += 4.0
                score -= 0.22 * opp_board_pressure
                if board_pressure_opp and opp_board_pressure >= 14.0:
                    score -= 6.0
                    if cur_opp_profile["best_roll"] >= 5:
                        score -= 8.0
                    if p < 0.80:
                        score -= 6.0
                if anti_bot3:
                    score -= anti_bot3_pressure
                    if lead >= 0 and cur_opp_profile["best_roll"] >= 4:
                        score -= 8.0
                    if lead >= -1 and p < 0.74:
                        score -= 4.0
                    if cur_profile["best_roll"] >= 2:
                        score -= 2.0

            else:
                nb_pos = nb.player_worker.position
                db = abs(best_loc[0] - my_pos[0]) + abs(best_loc[1] - my_pos[1])
                da = abs(best_loc[0] - nb_pos[0]) + abs(best_loc[1] - nb_pos[1])
                rat_progress = db - da
                rat_access_delta = self._rat_access_score(nb_pos, self.belief) - rat_access_now

                next_profile = self._carpet_profile(nb)
                next_opp_profile = self._opponent_carpet_profile(nb)
                next_corr = self._corridor_profile(nb, nb_pos)
                escape_bonus = self._dense_carpet_exit_score(board_obj, nb, my_pos, nb_pos, side_values)

                score += 1.8 * rat_progress * max(best_p, 0.25)
                score += 6.0 * rat_access_delta
                score += 0.8 * next_profile["line_value"]
                score += 1.5 * max(0.0, cur_opp_profile["line_value"] - next_opp_profile["line_value"])
                score += 1.3 * next_corr["value"]
                score += 2.0 * escape_bonus

                score += 2.2 * self._fresh_space_delta(board_obj, nb)
                score -= 2.8 * self._carpet_roam_penalty(nb)

                if self.turn - self.last_carpet_turn <= 3:
                    score += 4.0 * self._post_carpet_exit_value(nb)
                    if self._is_roaming_own_carpet(board_obj, nb_pos):
                        score -= 12.0

                if lead >= 4 and rat_access_delta <= 0.05 and next_profile["line_value"] <= cur_profile["line_value"] + 1.5:
                    score -= 8.0
                if lead >= 8 and dense_pressure_now >= 3.0 and side_values.get(mv.direction, -1e9) < best_side_val - 1.0:
                    score -= 6.0
                if board_pressure_opp and opp_board_pressure >= 14.0:
                    denial_delta = max(0.0, cur_opp_profile["line_value"] - next_opp_profile["line_value"])
                    if denial_delta <= 0.4 and rat_access_delta <= 0.12 and next_profile["best_roll"] < 4:
                        score -= 7.0
                    if opp_board_pressure >= 20.0 and denial_delta < 1.0 and next_profile["line_value"] <= cur_profile["line_value"] + 1.0:
                        score -= 5.0
                if anti_bot3:
                    denial_delta = max(0.0, cur_opp_profile["line_value"] - next_opp_profile["line_value"])
                    score += 1.2 * denial_delta
                    if (
                        anti_bot3_pressure >= 8.0
                        and rat_access_delta <= 0.08
                        and denial_delta <= 0.4
                        and next_profile["line_value"] <= cur_profile["line_value"] + 1.0
                    ):
                        score -= 5.0

            dest = self._move_destination(board_obj, mv)
            if dest is not None:
                score += self._novelty_bonus(dest)

            scored.append((score, mv))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    def _pick_safe_default(self, board_obj, ordered_moves):
        for mv in ordered_moves:
            if self._move_allowed(board_obj, mv):
                return mv
        return ordered_moves[0] if ordered_moves else None

    # =========================================================
    # SEARCH DISCIPLINE
    # =========================================================

    def _valid_search(self, board_obj, mv, belief) -> bool:
        if mv.move_type != enums.MoveType.SEARCH or mv.search_loc is None:
            return True

        x, y = mv.search_loc
        p = float(belief[y * BOARD_N + x])
        my_pos = board_obj.player_worker.position
        opp_pos = board_obj.opponent_worker.position
        lead = board_obj.player_worker.points - board_obj.opponent_worker.points
        turns = board_obj.player_worker.turns_left
        dist = abs(x - my_pos[0]) + abs(y - my_pos[1])
        opp_dist = abs(x - opp_pos[0]) + abs(y - opp_pos[1])
        peak = float(np.max(belief))
        peak_gap = self._belief_peak_gap(belief)
        entropy = self._belief_entropy(belief)

        # Stronger older-style search thresholds
        threshold = 0.64
        if lead <= -4:
            threshold = 0.44
        elif lead <= -2:
            threshold = 0.50
        elif lead <= 1:
            threshold = 0.56
        elif lead <= 3:
            threshold = 0.64
        elif lead >= 4:
            threshold = 0.69

        urgency = max(0.0, min(1.0, turns / ENDGAME_TURNS))
        threshold = threshold * urgency + 0.38 * (1.0 - urgency)

        if dist >= 3 and lead >= 0:
            threshold += 0.03
        if dist >= 4 and lead >= 0:
            threshold += 0.03
        if opp_dist + 1 < dist and p < 0.90:
            threshold += 0.04
        if lead >= 6:
            threshold += 0.03
        if lead >= 9 and turns >= 10:
            threshold += 0.04
        if peak >= 0.78:
            threshold -= 0.03
        if peak_gap >= 0.16:
            threshold -= 0.03
        if entropy <= 1.35:
            threshold -= 0.02
        threshold = max(0.38, min(0.84, threshold))

        if self.failed_search_cooldown > 0 and self.last_failed_search_loc == mv.search_loc:
            return False
        if self.failed_search_cooldown > 0 and p < (0.62 if turns <= 8 else 0.68):
            return False
        if peak_gap < 0.10 and entropy > 1.70 and p < 0.70:
            return False
        if p < threshold:
            return False

        recent_searches = sum(self.search_history)
        if recent_searches >= 2 and p < 0.76 and lead >= 0:
            return False
        if recent_searches >= 2 and lead < 0 and p < 0.68:
            return False
        if recent_searches >= 3 and p < 0.80:
            return False

        # Do not globally suppress search; only block if a genuinely strong carpet exists.
        if self._has_strong_carpet_move(board_obj, lead, p, peak):
            return False

        return True

    def _has_strong_carpet_move(self, board_obj, lead, p, peak) -> bool:
        my_pts = board_obj.player_worker.points
        best_gain = 0
        best_roll_pts = -10
        for mv in board_obj.get_valid_moves(exclude_search=True):
            if mv.move_type != enums.MoveType.CARPET or mv.roll_length < 2:
                continue
            nb = board_obj.forecast_move(mv)
            if nb is None:
                continue
            gain = nb.player_worker.points - my_pts
            best_gain = max(best_gain, gain)
            best_roll_pts = max(best_roll_pts, _cp(mv.roll_length))

        if best_roll_pts >= 6:
            return True
        if best_gain >= 10 and p < 0.82:
            return True
        if lead >= 0 and best_gain >= 6 and p < max(0.70, peak - 0.02):
            return True
        if lead >= -1 and best_gain >= 4 and p < 0.76:
            return True
        if lead >= 4 and best_gain >= 4 and p < max(0.66, peak - 0.02):
            return True
        if lead >= 2 and best_gain >= 6 and p < 0.72:
            return True
        if lead >= 3 and best_gain >= 10 and p < 0.80:
            return True
        return False

    # =========================================================
    # CARPET DISCIPLINE / DENIAL
    # =========================================================

    def _carpet_is_acceptable(self, board_obj, mv) -> bool:
        if mv.move_type != enums.MoveType.CARPET:
            return True

        length = mv.roll_length
        pts = _cp(length)
        lead = board_obj.player_worker.points - board_obj.opponent_worker.points
        turns = board_obj.player_worker.turns_left
        endgame = self._is_endgame(board_obj)
        threatened = self._opponent_corridor_pressure(board_obj) <= 3
        mature = self.corridor_age >= 3
        cashout_pressure = self._corridor_cashout_pressure(board_obj)
        anti_bot3 = self._is_bot3_like_opponent()
        anti_bot3_allow = False

        if anti_bot3 and pts >= 2 and lead >= 0:
            my_profile = self._carpet_profile(board_obj)
            opp_profile_board = self._opponent_carpet_profile(board_obj)
            anti_bot3_allow = (
                opp_profile_board["best_roll"] >= 4
                or opp_profile_board["line_value"] >= my_profile["line_value"] + 1.0
                or (turns <= 16 and opp_profile_board["line_value"] >= my_profile["line_value"] - 1.0)
            )
            if pts >= 4 and anti_bot3_allow:
                return True

        if length == 1:
            return endgame and self._is_desperate(board_obj)

        if pts >= 6:
            return True
        if lead >= 8 and pts >= 4:
            return True
        if lead >= 6 and turns <= 14 and (threatened or cashout_pressure >= 18.0) and pts >= 2:
            return True
        if lead >= 10 and turns <= 10 and pts >= 2:
            return True
        if cashout_pressure >= 24.0 and pts >= 2:
            return True

        if not endgame:
            two_move = self._best_carpet_in_two_moves(board_obj)
            if length <= 2 and two_move >= 6:
                if not (mature or threatened or anti_bot3_allow):
                    return False

            extension_gain = _cp(length + 1) - pts
            if extension_gain >= 4:
                if not (mature or threatened or anti_bot3_allow):
                    return False

            nb = board_obj.forecast_move(mv)
            if nb is not None:
                my_delta = self._carpet_profile(nb)["line_value"] - self._carpet_profile(board_obj)["line_value"]
                opp_delta = self._opponent_carpet_profile(nb)["line_value"] - self._opponent_carpet_profile(board_obj)["line_value"]
                if opp_delta > my_delta + 2.5:
                    return False

        return True

    # =========================================================
    # PROFILES / FEATURES
    # =========================================================

    def _carpet_profile(self, board_obj) -> dict:
        my_pts = board_obj.player_worker.points
        options = []

        try:
            moves = board_obj.get_valid_moves(exclude_search=True)
        except Exception:
            return {"best_gain": 0, "best_roll": 0, "line_value": 0.0, "long4_count": 0, "long5_count": 0}

        for mv in moves:
            if mv.move_type != enums.MoveType.CARPET:
                continue
            nb = board_obj.forecast_move(mv)
            if nb is None:
                continue
            gain = nb.player_worker.points - my_pts
            options.append((mv.roll_length, gain))

        if not options:
            return {"best_gain": 0, "best_roll": 0, "line_value": 0.0, "long4_count": 0, "long5_count": 0}

        options.sort(key=lambda item: (item[1], item[0]), reverse=True)
        best_roll, best_gain = options[0]
        second_gain = options[1][1] if len(options) > 1 else 0
        long4 = sum(1 for r, _ in options if r >= 4)
        long5 = sum(1 for r, _ in options if r >= 5)

        line_value = best_gain + 0.55 * second_gain + 2.5 * long4 + 3.8 * long5
        return {
            "best_gain": best_gain,
            "best_roll": best_roll,
            "line_value": line_value,
            "long4_count": long4,
            "long5_count": long5,
        }

    def _opponent_carpet_profile(self, board_obj) -> dict:
        try:
            rev = board_obj.get_copy()
            rev.reverse_perspective()
            return self._carpet_profile(rev)
        except Exception:
            return {"best_gain": 0, "best_roll": 0, "line_value": 0.0, "long4_count": 0, "long5_count": 0}

    def _best_carpet_in_two_moves(self, board_obj) -> int:
        my_pts = board_obj.player_worker.points
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
                gain = nb2.player_worker.points - my_pts
                best = max(best, gain)
        return best

    def _best_opponent_carpet_in_two_moves(self, board_obj) -> int:
        try:
            rev = board_obj.get_copy()
            rev.reverse_perspective()
            return self._best_carpet_in_two_moves(rev)
        except Exception:
            return 0

    def _corridor_profile(self, board_obj, pos) -> dict:
        pl = board_obj.player_worker.position
        opp = board_obj.opponent_worker.position
        best_run = 0
        best_primed = 0

        for direction in enums.Direction:
            loc = pos
            run = 0
            primed = 0
            for _ in range(7):
                loc = enums.loc_after_direction(loc, direction)
                x, y = loc
                if not (0 <= x < BOARD_N and 0 <= y < BOARD_N):
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
                best_run = run
                best_primed = primed

        value = best_run + 0.8 * best_primed
        if best_run >= 4:
            value += 3.0
        if best_run >= 5:
            value += 4.0
        return {"best_run": best_run, "best_primed": best_primed, "value": value}

    def _corridor_profile_for_opponent(self, board_obj) -> dict:
        try:
            rev = board_obj.get_copy()
            rev.reverse_perspective()
            return self._corridor_profile(rev, rev.player_worker.position)
        except Exception:
            return {"best_run": 0, "best_primed": 0, "value": 0.0}

    def _setup_plan_profile(self, board_obj, depth=2) -> dict:
        profile = self._carpet_profile(board_obj)
        corridor = self._corridor_profile(board_obj, board_obj.player_worker.position)
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

        candidates = []
        for mv in moves:
            if mv.move_type == enums.MoveType.CARPET and mv.roll_length < 4:
                continue
            if mv.move_type not in (enums.MoveType.PLAIN, enums.MoveType.PRIME, enums.MoveType.CARPET):
                continue
            pri = {enums.MoveType.PRIME: 3, enums.MoveType.PLAIN: 2}.get(mv.move_type, 1 + mv.roll_length)
            candidates.append((pri, mv))

        candidates.sort(key=lambda item: item[0], reverse=True)

        best_val = base_value
        best_roll = profile["best_roll"]
        my_pts = board_obj.player_worker.points

        for _, mv in candidates[:6]:
            nb = board_obj.forecast_move(mv)
            if nb is None:
                continue
            child = self._setup_plan_profile(nb, depth - 1)
            gain = nb.player_worker.points - my_pts
            value = child["value"] + 2.5 * gain

            if mv.move_type == enums.MoveType.PRIME:
                value += 4.0
            elif mv.move_type == enums.MoveType.PLAIN:
                value += 1.5
            elif mv.move_type == enums.MoveType.CARPET and mv.roll_length >= 4:
                value += 6.0 + 2.0 * mv.roll_length

            child_roll = max(child["best_roll"], mv.roll_length if mv.move_type == enums.MoveType.CARPET else 0)
            if value > best_val or (abs(value - best_val) < 1e-6 and child_roll > best_roll):
                best_val = value
                best_roll = child_roll

        return {"value": best_val, "best_roll": best_roll}

    # =========================================================
    # BELIEF HELPERS
    # =========================================================

    def _top_k_belief(self, belief: np.ndarray, k=10):
        idx = np.argsort(belief)[-k:]
        out = np.zeros_like(belief)
        out[idx] = belief[idx]
        s = out.sum()
        return out / s if s > 0 else belief

    def _best_search_cell(self):
        idx = int(np.argmax(self.belief))
        return (idx % BOARD_N, idx // BOARD_N), float(self.belief[idx])

    def _belief_entropy(self, belief: np.ndarray) -> float:
        x = np.clip(belief, 1e-12, 1.0)
        return float(-np.sum(x * np.log(x)))

    def _belief_peak_gap(self, belief: np.ndarray) -> float:
        if belief.size < 2:
            return float(np.max(belief))
        top = np.partition(belief, -2)[-2:]
        return float(top[-1] - top[-2])

    def _expected_rat_distance_from(self, pos, belief: np.ndarray, k: int = 14) -> float:
        total = 0.0
        total_p = 0.0
        for idx in np.argsort(belief)[-k:]:
            p = float(belief[idx])
            if p <= 0:
                continue
            x, y = idx % BOARD_N, idx // BOARD_N
            total += p * (abs(x - pos[0]) + abs(y - pos[1]))
            total_p += p
        return total / total_p if total_p > 0 else 6.0

    def _belief_local_mass(self, pos, belief: np.ndarray, radius: int = 1, k: int = 14) -> float:
        total = 0.0
        for idx in np.argsort(belief)[-k:]:
            p = float(belief[idx])
            if p <= 0:
                continue
            x, y = idx % BOARD_N, idx // BOARD_N
            if abs(x - pos[0]) + abs(y - pos[1]) <= radius:
                total += p
        return total

    def _rat_access_score(self, pos, belief: np.ndarray) -> float:
        near1 = self._belief_local_mass(pos, belief, radius=1, k=14)
        near2 = self._belief_local_mass(pos, belief, radius=2, k=14)
        expected = self._expected_rat_distance_from(pos, belief, k=14)
        return 3.0 * near1 + 1.4 * near2 + 2.4 / (1.0 + expected)

    def _race_value_from_belief(self, my_pos, opp_pos, belief) -> float:
        idxs = np.argsort(belief)[-6:]
        val = 0.0
        for idx in idxs:
            p = float(belief[idx])
            x, y = idx % BOARD_N, idx // BOARD_N
            d_me = abs(x - my_pos[0]) + abs(y - my_pos[1])
            d_opp = abs(x - opp_pos[0]) + abs(y - opp_pos[1])
            val += p * (d_opp - d_me)
        return val

    def _rat_zone_control(self, board_obj, belief) -> float:
        my_pos = board_obj.player_worker.position
        opp_pos = board_obj.opponent_worker.position
        total = 0.0
        for idx in np.argsort(belief)[-8:]:
            p = float(belief[idx])
            x, y = idx % BOARD_N, idx // BOARD_N
            d_me = abs(x - my_pos[0]) + abs(y - my_pos[1])
            d_opp = abs(x - opp_pos[0]) + abs(y - opp_pos[1])
            total += p * (d_opp - d_me)
        return total

    # =========================================================
    # SMALL POSITIONAL HELPERS
    # =========================================================

    def _mobility_value(self, board_obj) -> float:
        try:
            mine = len(board_obj.get_valid_moves(exclude_search=True))
        except Exception:
            mine = 0
        try:
            rev = board_obj.get_copy()
            rev.reverse_perspective()
            opp = len(rev.get_valid_moves(exclude_search=True))
        except Exception:
            opp = 0
        return float(mine - 0.7 * opp)

    def _local_control(self, board_obj) -> float:
        my_pos = board_obj.player_worker.position
        opp_pos = board_obj.opponent_worker.position
        total = 0.0
        for x in range(BOARD_N):
            for y in range(BOARD_N):
                d_me = abs(x - my_pos[0]) + abs(y - my_pos[1])
                d_opp = abs(x - opp_pos[0]) + abs(y - opp_pos[1])
                total += 0.12 * (d_opp - d_me)
        return total

    def _direction_between(self, start, dest):
        dx = dest[0] - start[0]
        dy = dest[1] - start[1]
        if dx > 0 and dy == 0:
            return enums.Direction.RIGHT
        if dx < 0 and dy == 0:
            return enums.Direction.LEFT
        if dy > 0 and dx == 0:
            return enums.Direction.DOWN
        if dy < 0 and dx == 0:
            return enums.Direction.UP
        return None

    def _anti_loop_eval(self, my_pos) -> float:
        if not self.position_history:
            return 0.0
        hist = list(self.position_history)
        repeats = sum(1 for p in hist if p == my_pos)
        oscillate = 0
        if len(hist) >= 4:
            oscillate = sum(1 for i in range(len(hist) - 2) if hist[i] == hist[i + 2])
        penalty = -2.5 * max(0, repeats - 2) - 1.5 * max(0, oscillate - 2)
        if self.no_gain_streak >= 4:
            penalty -= 1.5
        if self.no_gain_streak >= 7:
            penalty -= 2.5
        return penalty

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

    def _count_cells(self, board_obj, cell_type) -> int:
        total = 0
        for x in range(BOARD_N):
            for y in range(BOARD_N):
                try:
                    if board_obj.get_cell((x, y)) == cell_type:
                        total += 1
                except Exception:
                    pass
        return total

    def _is_endgame(self, board_obj) -> bool:
        return board_obj.player_worker.turns_left <= ENDGAME_TURNS

    def _is_desperate(self, board_obj) -> bool:
        lead = board_obj.player_worker.points - board_obj.opponent_worker.points
        return lead <= -4 and board_obj.player_worker.turns_left <= 20

    def _carpet_neighbor_count(self, board_obj, pos) -> int:
        total = 0
        for direction in enums.Direction:
            loc = enums.loc_after_direction(pos, direction)
            x, y = loc
            if not (0 <= x < BOARD_N and 0 <= y < BOARD_N):
                continue
            try:
                if board_obj.get_cell(loc) == enums.Cell.CARPET:
                    total += 1
            except Exception:
                pass
        return total

    def _dense_carpet_pressure_at(self, board_obj, pos) -> float:
        pressure = 0.0
        try:
            if board_obj.get_cell(pos) == enums.Cell.CARPET:
                pressure += 1.6
        except Exception:
            return 0.0

        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                x = pos[0] + dx
                y = pos[1] + dy
                if not (0 <= x < BOARD_N and 0 <= y < BOARD_N):
                    continue
                man = abs(dx) + abs(dy)
                weight = 1.0 if man <= 1 else (0.65 if man == 2 else 0.35)
                try:
                    cell = board_obj.get_cell((x, y))
                except Exception:
                    continue
                if cell == enums.Cell.CARPET:
                    pressure += weight
                elif cell == enums.Cell.SPACE:
                    pressure -= 0.18 * weight
                elif cell == enums.Cell.PRIMED:
                    pressure -= 0.08 * weight

        return pressure

    def _directional_uncarpeted_value(self, board_obj, pos, direction) -> float:
        total = 0.0
        for x in range(BOARD_N):
            for y in range(BOARD_N):
                dx = x - pos[0]
                dy = y - pos[1]
                if dx == 0 and dy == 0:
                    continue

                if direction == enums.Direction.UP and dy >= 0:
                    continue
                if direction == enums.Direction.DOWN and dy <= 0:
                    continue
                if direction == enums.Direction.LEFT and dx >= 0:
                    continue
                if direction == enums.Direction.RIGHT and dx <= 0:
                    continue

                dist = abs(dx) + abs(dy)
                weight = 1.0 / (1.0 + 0.30 * max(0, dist - 1))
                if direction in (enums.Direction.LEFT, enums.Direction.RIGHT) and abs(dy) <= 1:
                    weight *= 1.15
                if direction in (enums.Direction.UP, enums.Direction.DOWN) and abs(dx) <= 1:
                    weight *= 1.15

                try:
                    cell = board_obj.get_cell((x, y))
                except Exception:
                    continue

                if cell == enums.Cell.SPACE:
                    total += 1.3 * weight
                elif cell == enums.Cell.PRIMED:
                    total += 0.85 * weight
                elif cell == enums.Cell.CARPET:
                    total -= 0.75 * weight
                elif cell == enums.Cell.BLOCKED:
                    total -= 0.20 * weight

        return total

    def _best_uncarpeted_side(self, board_obj, pos):
        values = {}
        best_dir = None
        best_val = -1e18
        for direction in enums.Direction:
            val = self._directional_uncarpeted_value(board_obj, pos, direction)
            values[direction] = val
            if val > best_val:
                best_dir = direction
                best_val = val
        return best_dir, best_val, values

    def _dense_carpet_exit_score(self, before_board, after_board, start_pos, dest_pos, side_values=None) -> float:
        direction = self._direction_between(start_pos, dest_pos)
        if direction is None:
            return 0.0

        before_pressure = self._dense_carpet_pressure_at(before_board, start_pos)
        after_pressure = self._dense_carpet_pressure_at(after_board, dest_pos)

        if side_values is None:
            best_dir, best_side_val, side_values = self._best_uncarpeted_side(before_board, start_pos)
        else:
            best_dir = max(side_values, key=side_values.get) if side_values else None
            best_side_val = side_values.get(best_dir, 0.0) if best_dir is not None else 0.0

        dir_val = side_values.get(direction, 0.0) if side_values else 0.0
        score = 0.0

        if before_pressure >= 2.8:
            score += 1.8 * (before_pressure - after_pressure)
            score += 0.28 * dir_val
            if direction == best_dir:
                score += 3.0 + 0.16 * max(0.0, best_side_val)
        else:
            score += 0.55 * max(0.0, before_pressure - after_pressure)

        try:
            before_cell = before_board.get_cell(start_pos)
            after_cell = after_board.get_cell(dest_pos)
        except Exception:
            before_cell = None
            after_cell = None

        if before_cell == enums.Cell.CARPET and after_cell != enums.Cell.CARPET:
            score += 2.6
        if after_cell == enums.Cell.CARPET and before_pressure >= 3.2:
            score -= 2.8

        return score

    def _is_roaming_own_carpet(self, board_obj, pos) -> bool:
        try:
            if board_obj.get_cell(pos) != enums.Cell.CARPET:
                return False
        except Exception:
            return False

        recent = list(self.position_history)[-6:]
        same_zone = sum(
            1 for p in recent
            if abs(p[0] - pos[0]) <= 1 and abs(p[1] - pos[1]) <= 1
        )
        carpet_neighbors = self._carpet_neighbor_count(board_obj, pos)
        return same_zone >= 4 or carpet_neighbors >= 3

    def _carpet_roam_penalty(self, board_obj) -> float:
        pos = board_obj.player_worker.position
        if not self._is_roaming_own_carpet(board_obj, pos):
            return 0.0

        penalty = 1.8
        recent = list(self.position_history)[-8:]
        unique_recent = len(set(recent)) if recent else 0
        if unique_recent <= 4:
            penalty += 1.4
        if unique_recent <= 3:
            penalty += 1.8
        if self.same_tile_streak >= 1:
            penalty += 1.5
        if self.same_tile_streak >= 2:
            penalty += 2.5
        return penalty

    def _fresh_space_value(self, board_obj) -> float:
        pos = board_obj.player_worker.position
        total = 0.0
        for direction in enums.Direction:
            loc = enums.loc_after_direction(pos, direction)
            x, y = loc
            if not (0 <= x < BOARD_N and 0 <= y < BOARD_N):
                continue
            if loc == board_obj.opponent_worker.position:
                continue
            try:
                cell = board_obj.get_cell(loc)
            except Exception:
                continue
            if cell == enums.Cell.SPACE:
                total += 2.0
            elif cell == enums.Cell.PRIMED:
                total += 1.2
            elif cell == enums.Cell.CARPET:
                total -= 0.8
        return total

    def _fresh_space_delta(self, before_board, after_board) -> float:
        return self._fresh_space_value(after_board) - self._fresh_space_value(before_board)

    def _post_carpet_exit_value(self, board_obj) -> float:
        pos = board_obj.player_worker.position
        total = 0.0
        for direction in enums.Direction:
            loc = enums.loc_after_direction(pos, direction)
            x, y = loc
            if not (0 <= x < BOARD_N and 0 <= y < BOARD_N):
                continue
            if loc == board_obj.opponent_worker.position:
                continue
            try:
                cell = board_obj.get_cell(loc)
            except Exception:
                continue
            if cell == enums.Cell.SPACE:
                total += 2.6
            elif cell == enums.Cell.PRIMED:
                total += 1.5
            elif cell == enums.Cell.CARPET:
                total -= 1.4
        return total
