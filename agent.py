"""
Expectiminimax search with alpha-beta pruning, iterative deepening,
transposition table, move ordering, PVS + LMR, and a Carrie-style
cell-potential heuristic.

Design notes
------------
  * The chance nodes (rat position, sensor noise) are collapsed into an
    HMM belief. Search EV is computed analytically from the belief:
    EV(search @ cell) = 6 * p - 2.
  * Rat movement through tree depth is modelled by propagating the
    belief one step per turn via the transition matrix.
  * Opponent searches inside the tree fold EXPECTED post-search belief
    into the child (so min nodes get credit for information + hit).
  * Heuristic: a cell-potential map (prime-and-carpet marginal value of
    every cell) + attributed chain ownership (who races to carpet
    which existing run) + search EV + rat approach bias. The feature
    set is race- and shape-aware, matching the spec's Carrie
    description.
  * NEGAMAX framing: value is always from the perspective of the
    player about to move; we negate on recursion.
"""

import time
from collections import deque
from typing import Optional, Tuple

import numpy as np

from game.move import Move
from game.enums import MoveType, Direction, Cell, CARPET_POINTS_TABLE, BOARD_SIZE


NEG_INF = -1e9
POS_INF = 1e9

# Transposition table flags
TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2


class TimeoutSignal(Exception):
    pass


# -----------------------------------------------------------------------------
# Heuristic weights. These are DEFAULTS; override via HeuristicConfig.
# -----------------------------------------------------------------------------
# The weight "schema" was reshaped after we discovered the previous heuristic
# was double-counting carpet-ready chains in prime_run_potent. The new
# features (immediate_carpet, chain_ownership, cell_potential) are additive
# and avoid that overlap.
W_POINT_DIFF         = 1.00   # direct score differential
W_IMMEDIATE_CARPET   = 1.00   # realizable-now best carpet. Iteration-6 tournament
                              # lost carpet races (match 5: 9 vs 4; match 6:
                              # 7 vs 1). Bumped 0.85 -> 1.00 to grab carpets
                              # more aggressively. 8-seed sweep confirms
                              # 1.00/chain=0.70 ties current on margin
                              # (+17.88 vs +17.75) with more carpet focus.
W_CHAIN_OWNERSHIP    = 0.70   # race-attributed primed chains (not carpet-ready now)
                              # Pulled back from 0.85 to 0.70 to make room
                              # for W_IMMEDIATE_CARPET=1.00. Sweep showed
                              # imm=1.00/chain=0.85 regressed to +6.62
                              # while imm=1.00/chain=0.70 held at +17.88 —
                              # these weights are partially substitutive.
W_CELL_POTENTIAL     = 0.25   # Carrie-style: potential of priming any cell, distance decayed
W_SEARCH_EV          = 0.90   # best +EV search available
W_RAT_APPROACH       = 0.06   # walk toward highest-belief cell
W_OPP_THREAT         = 0.10   # iter-10: relaxed from 0.25. Sweep showed
                              # over-weighting opp_threat made us too
                              # defensive and sacrificed our own setups.
                              # E_opp10 variant: vs george +40.4 (baseline
                              # +31.6, +8.8 gain) vs albertlite +31.1
                              # (baseline +14.5, +16.6 gain). 8W-0L both.
W_LEAD_PRESERVATION  = 0.15   # scale point_diff when ahead with few turns left
W_MOBILITY           = 0.15   # penalize dead-ends (mob<=1) only; 0.30 was too
                              # aggressive and discouraged chain-building primes


class HeuristicConfig:
    """Bundle of heuristic weights so we can run parameter sweeps without
    patching module globals (which would break multiprocessing)."""
    __slots__ = (
        "point_diff", "immediate_carpet", "chain_ownership",
        "cell_potential", "search_ev", "rat_approach",
        "opp_threat", "lead_preservation", "mobility",
        # Legacy aliases kept so tuple-based sweep.py / validate.py keep
        # working. The HeuristicConfig(*tuple) positional pattern uses
        # (pd, cr, prp, ext, sev, hbd).
        "carpet_ready", "prime_run_potent", "extension_potent",
        "high_belief_dist",
    )

    def __init__(self,
                 point_diff=W_POINT_DIFF,
                 immediate_carpet=None,       # legacy: carpet_ready
                 chain_ownership=None,        # legacy: prime_run_potent * 2.5
                 cell_potential=None,         # legacy: extension_potent
                 search_ev=W_SEARCH_EV,
                 rat_approach=None,           # legacy: high_belief_dist
                 opp_threat=W_OPP_THREAT,
                 lead_preservation=W_LEAD_PRESERVATION,
                 mobility=W_MOBILITY,
                 *,
                 carpet_ready=None,
                 prime_run_potent=None,
                 extension_potent=None,
                 high_belief_dist=None):
        self.point_diff = point_diff

        # Map legacy names to new fields.
        if immediate_carpet is None:
            immediate_carpet = carpet_ready if carpet_ready is not None else W_IMMEDIATE_CARPET
        if chain_ownership is None:
            chain_ownership = (prime_run_potent * 2.5) if prime_run_potent is not None else W_CHAIN_OWNERSHIP
        if cell_potential is None:
            cell_potential = extension_potent if extension_potent is not None else W_CELL_POTENTIAL
        if rat_approach is None:
            rat_approach = high_belief_dist if high_belief_dist is not None else W_RAT_APPROACH

        self.immediate_carpet = immediate_carpet
        self.chain_ownership = chain_ownership
        self.cell_potential = cell_potential
        self.search_ev = search_ev
        self.rat_approach = rat_approach
        self.opp_threat = opp_threat
        self.lead_preservation = lead_preservation
        self.mobility = mobility

        # Populate legacy knob fields so tuple-roundtrip works.
        self.carpet_ready = immediate_carpet
        self.prime_run_potent = chain_ownership / 2.5
        self.extension_potent = cell_potential
        self.high_belief_dist = rat_approach

    def as_tuple(self):
        """Legacy tuple shape: (pd, cr, prp, ext, sev, hbd)."""
        return (self.point_diff, self.carpet_ready, self.prime_run_potent,
                self.extension_potent, self.search_ev, self.high_belief_dist)

    def __repr__(self):
        return (f"HeuristicConfig(pd={self.point_diff:.2f}, "
                f"imm={self.immediate_carpet:.2f}, "
                f"chain={self.chain_ownership:.2f}, "
                f"pot={self.cell_potential:.2f}, "
                f"sev={self.search_ev:.2f}, rat={self.rat_approach:.2f}, "
                f"thr={self.opp_threat:.2f})")


_DEFAULT_CONFIG = HeuristicConfig()


# =============================================================================
# Bitboard helpers
# =============================================================================
FULL_MASK = 0xFFFFFFFFFFFFFFFF


def _space_mask(board):
    """Bitboard of SPACE cells (not primed / carpeted / blocked)."""
    return FULL_MASK & ~(board._primed_mask | board._carpet_mask | board._blocked_mask)


# =============================================================================
# Prime runs (maximal contiguous primed lines, length >= 2)
# =============================================================================
def _prime_runs(board):
    runs = []
    for y in range(BOARD_SIZE):
        x = 0
        while x < BOARD_SIZE:
            if board.get_cell((x, y)) == Cell.PRIMED:
                start_x = x
                while x < BOARD_SIZE and board.get_cell((x, y)) == Cell.PRIMED:
                    x += 1
                length = x - start_x
                if length >= 2:
                    runs.append((length, (start_x, y), (x - 1, y)))
            else:
                x += 1
    for x in range(BOARD_SIZE):
        y = 0
        while y < BOARD_SIZE:
            if board.get_cell((x, y)) == Cell.PRIMED:
                start_y = y
                while y < BOARD_SIZE and board.get_cell((x, y)) == Cell.PRIMED:
                    y += 1
                length = y - start_y
                if length >= 2:
                    runs.append((length, (x, start_y), (x, y - 1)))
            else:
                y += 1
    return runs


# =============================================================================
# BFS distance map (precomputed all-pairs, cached per blocked_mask)
# =============================================================================
# blocked_mask is immutable across a game, so the full 64x64 distance matrix
# is computed exactly once the first time a game's blocked_mask is seen.
# Each subsequent _bfs_dist() call is a single numpy row lookup.
_BFS_CACHE = {}


def _bfs_all_from(blocked_mask):
    """Return (64, 64) int32 distance matrix: [si, ti] is dist si->ti.
    BLOCKED cells are impassable; all other cells (PRIMED, CARPET, SPACE)
    are passable (relaxation -- safe for a heuristic)."""
    cached = _BFS_CACHE.get(blocked_mask)
    if cached is not None:
        return cached
    if len(_BFS_CACHE) > 64:
        _BFS_CACHE.clear()

    dist = np.full((64, 64), 99, dtype=np.int32)
    for si in range(64):
        if blocked_mask & (1 << si):
            continue
        local = np.full(64, 99, dtype=np.int32)
        local[si] = 0
        q = deque([si])
        while q:
            i = q.popleft()
            d_i = int(local[i])
            x = i & 7
            y = i >> 3
            # unrolled 4-neighbour expansion
            if x + 1 < BOARD_SIZE:
                ni = i + 1
                if not (blocked_mask & (1 << ni)) and local[ni] > d_i + 1:
                    local[ni] = d_i + 1
                    q.append(ni)
            if x - 1 >= 0:
                ni = i - 1
                if not (blocked_mask & (1 << ni)) and local[ni] > d_i + 1:
                    local[ni] = d_i + 1
                    q.append(ni)
            if y + 1 < BOARD_SIZE:
                ni = i + BOARD_SIZE
                if not (blocked_mask & (1 << ni)) and local[ni] > d_i + 1:
                    local[ni] = d_i + 1
                    q.append(ni)
            if y - 1 >= 0:
                ni = i - BOARD_SIZE
                if not (blocked_mask & (1 << ni)) and local[ni] > d_i + 1:
                    local[ni] = d_i + 1
                    q.append(ni)
        dist[si] = local
    _BFS_CACHE[blocked_mask] = dist
    return dist


def _bfs_dist(board, start):
    """BFS distance from `start` -- O(1) lookup after first call per game."""
    si = start[1] * BOARD_SIZE + start[0]
    return _bfs_all_from(board._blocked_mask)[si]


# =============================================================================
# Cell potential: marginal value of priming each cell (vectorised + cached)
# =============================================================================
# Carpet-points lookup, flattened to a numpy array indexed by run length 0..7.
_CPT_ARR = np.zeros(8, dtype=np.float64)
for _k, _v in CARPET_POINTS_TABLE.items():
    if 0 <= int(_k) <= 7:
        _CPT_ARR[int(_k)] = float(_v)

# Cache keyed by (primed_mask, carpet_mask, blocked_mask) -- same mask tuple
# always yields the same potential vector. Cheap to compute but we hit it at
# every leaf, so memoisation is a big win in deep iterative-deepening runs.
_POT_CACHE = {}


def _mask_to_grid(mask):
    """64-bit int -> (8,8) bool grid; grid[y, x] == bit at y*8+x."""
    bits = np.fromiter(((mask >> i) & 1 for i in range(64)),
                       dtype=np.uint8, count=64)
    return bits.reshape(BOARD_SIZE, BOARD_SIZE).astype(bool)


def _cell_potential_vec(board):
    """
    For each cell c, compute the marginal value of priming c:
        potential(c) = 1 + max(carpet_h, carpet_v)
    Only SPACE cells get nonzero potential.

    Returns a (64,) float64 numpy array.
    """
    primed = board._primed_mask
    carpet = board._carpet_mask
    blocked = board._blocked_mask
    key = (primed, carpet, blocked)
    cached = _POT_CACHE.get(key)
    if cached is not None:
        return cached
    if len(_POT_CACHE) > 20000:
        _POT_CACHE.clear()

    primed_grid = _mask_to_grid(primed)
    # Space = not primed, not carpet, not blocked
    occupied_grid = _mask_to_grid(primed | carpet | blocked)
    space_grid = ~occupied_grid

    # Run-length of consecutive primed cells immediately preceding each
    # position, in each of 4 directions. Each loop is 7 vectorised steps
    # over an 8-column/row slice.
    L_left = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
    for x in range(1, BOARD_SIZE):
        L_left[:, x] = np.where(primed_grid[:, x - 1], L_left[:, x - 1] + 1, 0)
    L_right = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
    for x in range(BOARD_SIZE - 2, -1, -1):
        L_right[:, x] = np.where(primed_grid[:, x + 1], L_right[:, x + 1] + 1, 0)
    L_up = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
    for y in range(1, BOARD_SIZE):
        L_up[y, :] = np.where(primed_grid[y - 1, :], L_up[y - 1, :] + 1, 0)
    L_down = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
    for y in range(BOARD_SIZE - 2, -1, -1):
        L_down[y, :] = np.where(primed_grid[y + 1, :], L_down[y + 1, :] + 1, 0)

    L_h = L_left + L_right + 1
    L_v = L_up + L_down + 1
    L_h_clip = np.minimum(L_h, 7)
    L_v_clip = np.minimum(L_v, 7)
    carpet_h = np.where(L_h >= 2, _CPT_ARR[L_h_clip], 0.0)
    carpet_v = np.where(L_v >= 2, _CPT_ARR[L_v_clip], 0.0)
    best_carpet = np.maximum(carpet_h, carpet_v)
    pot_grid = np.where(space_grid, 1.0 + best_carpet, 0.0)
    pot_vec = pot_grid.ravel().astype(np.float64, copy=False)

    _POT_CACHE[key] = pot_vec
    return pot_vec


def _potential_for_worker(pot_vec, dist_map, turns_left):
    """
    Sum cell-potential contributions, distance-decayed from the worker
    and clipped to cells reachable within `turns_left`.
    decay(d) = max(0, 1 - d / 6).
    """
    decay = np.maximum(0.0, 1.0 - dist_map / 6.0)
    reachable = (dist_map <= turns_left)
    return float(np.sum(pot_vec * decay * reachable))


# =============================================================================
# Attributed chain value (race-to-carpet, with double-count fix)
# =============================================================================
def _attributed_chain_value(board, my_dist, op_dist,
                            my_carpet_now, op_carpet_now):
    """
    For each prime run, attribute its carpet value to whichever worker
    can reach an endpoint sooner. Skip runs that either player is about
    to carpet RIGHT NOW (those are already credited in immediate_carpet,
    avoiding double-count).

    Returns (my_attrib, op_attrib).
    """
    my_total = 0.0
    op_total = 0.0

    for length, ep_a, ep_b in _prime_runs(board):
        capped = min(length, 7)
        carpet_value = CARPET_POINTS_TABLE.get(capped, 0)
        if carpet_value <= 0:
            continue

        my_d = min(int(my_dist[ep_a[1] * BOARD_SIZE + ep_a[0]]),
                   int(my_dist[ep_b[1] * BOARD_SIZE + ep_b[0]]))
        op_d = min(int(op_dist[ep_a[1] * BOARD_SIZE + ep_a[0]]),
                   int(op_dist[ep_b[1] * BOARD_SIZE + ep_b[0]]))

        # Skip if carpet-ready (already in immediate_carpet).
        if my_d <= 1 and carpet_value <= my_carpet_now:
            continue
        if op_d <= 1 and carpet_value <= op_carpet_now:
            continue

        def _decay(d):
            return max(0.15, 1.0 - d / 8.0)

        if my_d < op_d:
            my_total += carpet_value * _decay(my_d)
        elif op_d < my_d:
            op_total += carpet_value * _decay(op_d)
        else:
            split = carpet_value * 0.5 * _decay(my_d)
            my_total += split
            op_total += split
    return my_total, op_total


# =============================================================================
# Search EV / belief helpers
# =============================================================================
def _best_search_ev(belief):
    p_max = float(np.max(belief))
    return max(0.0, 6.0 * p_max - 2.0)


def _distance_to_high_belief(belief, worker_pos):
    """Manhattan distance from the worker to the top-3 belief centroid.
    Smoother than argmax when the rat is between two cells."""
    top3 = np.argpartition(belief, -3)[-3:]
    weights = belief[top3]
    total = weights.sum()
    if total < 1e-12:
        idx = int(np.argmax(belief))
        rx, ry = idx % BOARD_SIZE, idx // BOARD_SIZE
        return abs(rx - worker_pos[0]) + abs(ry - worker_pos[1])
    rxs = (top3 % BOARD_SIZE).astype(np.float64)
    rys = (top3 // BOARD_SIZE).astype(np.float64)
    mean_x = float((rxs * weights).sum() / total)
    mean_y = float((rys * weights).sum() / total)
    return abs(mean_x - worker_pos[0]) + abs(mean_y - worker_pos[1])


def _mobility_score(board, pos):
    """Count 4-neighbors passable to WALKING. Primed and blocked cells are
    both barriers (see Board.is_cell_blocked). Carpet and space are OK.

    Range 0-4. Low values indicate dead-end corners where the agent may
    be forced into bad searches or wasted plain moves."""
    x0, y0 = pos
    primed = board._primed_mask
    blocked = board._blocked_mask
    bad = primed | blocked
    count = 0
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x0 + dx, y0 + dy
        if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
            bit = 1 << (ny * BOARD_SIZE + nx)
            if not (bad & bit):
                count += 1
    return count


def _best_immediate_carpet(board, enemy=False):
    best = 0
    for m in board.get_valid_moves(enemy=enemy, exclude_search=True):
        if m.move_type == MoveType.CARPET and m.roll_length >= 2:
            pts = CARPET_POINTS_TABLE[m.roll_length]
            if pts > best:
                best = pts
    return best


# =============================================================================
# MAIN HEURISTIC
# =============================================================================
def heuristic_eval(board, belief, cfg=_DEFAULT_CONFIG):
    """Evaluate from the perspective of board.player_worker."""
    my_pos = board.player_worker.get_location()
    op_pos = board.opponent_worker.get_location()

    point_diff = board.player_worker.points - board.opponent_worker.points

    my_carpet_now = _best_immediate_carpet(board, enemy=False)
    op_carpet_now = _best_immediate_carpet(board, enemy=True)

    my_dist = _bfs_dist(board, my_pos)
    op_dist = _bfs_dist(board, op_pos)

    my_chains, op_chains = _attributed_chain_value(
        board, my_dist, op_dist, my_carpet_now, op_carpet_now
    )

    pot_vec = _cell_potential_vec(board)
    turns_left = min(board.player_worker.turns_left,
                     board.opponent_worker.turns_left)
    my_pot = _potential_for_worker(pot_vec, my_dist, turns_left)
    op_pot = _potential_for_worker(pot_vec, op_dist, turns_left)

    search_ev = _best_search_ev(belief)
    rat_dist  = _distance_to_high_belief(belief, my_pos)

    # Future-value discount in the endgame.
    endgame_factor = max(0.1, min(1.0, turns_left / 6.0))

    # Lead preservation: the last few turns amplify point_diff so we
    # don't trade away leads. Capped so it can't overwhelm big swings.
    urgency = max(0.0, 1.0 - turns_left / 10.0)
    lead_bonus = (cfg.lead_preservation * urgency
                  * np.sign(point_diff) * min(abs(point_diff), 20))

    # NOTE: We tried a mobility / corner-lockout term to address the
    # corner-spam behavior seen in tournament loss logs. It was net
    # negative on the 20-game benchmark (19/1 +17.4 without -> 17/2/1
    # +11.7 with W=0.15) because it interfered with chain-building
    # primes. The root cause of corner-spam (permissive search
    # threshold) is already fixed, so the term is unnecessary. The
    # cfg.mobility weight and _mobility_score helper are kept for
    # potential future experiments.
    _ = cfg.mobility  # keep attribute live for sweeps

    value = (
        cfg.point_diff       * point_diff
      + cfg.immediate_carpet * (my_carpet_now - op_carpet_now)
      + cfg.chain_ownership  * endgame_factor * (my_chains - op_chains)
      + cfg.cell_potential   * endgame_factor * (my_pot - op_pot)
      + cfg.search_ev        * search_ev
      - cfg.rat_approach     * rat_dist
      - cfg.opp_threat       * op_carpet_now
      + lead_bonus
    )
    return float(value)


def terminal_value(board):
    diff = board.player_worker.points - board.opponent_worker.points
    if diff > 0:
        return 100.0 + diff
    elif diff < 0:
        return -100.0 + diff
    else:
        return 0.0


# =============================================================================
# Move filtering and ordering
# =============================================================================
def _search_ev_at(belief, loc):
    idx = loc[1] * BOARD_SIZE + loc[0]
    p = float(belief[idx])
    return 6.0 * p - 2.0


# Search thresholds have been raised THREE times now:
#   0.30 -> 1.50 (p_max 0.38 -> 0.58) after first set of tournament losses.
#   1.50 -> 3.00 (p_max 0.58 -> 0.83) after iteration-5 tournament run
#     showed 0/19 hit rate across 4 losses (bleeding ~9.5 pts/game).
#   Desperate 0.50 -> 3.00 after iteration-6 match (6) showed the
#     desperate carve-out firing 3 times at 0/3 hits, cascading a -21
#     loss. If our belief is systematically overconfident, being
#     behind doesn't suddenly make -2 EV a good bet. Kept the name
#     DESPERATE for code clarity but functionally disabled it — the
#     desperate flag now has no effect on search thresholds.
_ROOT_SEARCH_THRESHOLD_DEFAULT = 2.00      # iter-9 experiment: iter-8 had
                                           # 0/8 misses in tournament at 3.00
                                           # but also 0/8 hits vs Carrie while
                                           # she got 5 hits in match 7. 3.00
                                           # was over-conservative; try 2.00
                                           # (p_max > 0.667) for Carrie parity.
_ROOT_SEARCH_THRESHOLD_DESPERATE = 2.00    # keep symmetric with default
_ROOT_SEARCH_THRESHOLD_CAUTIOUS = 3.00     # ahead + late: still cautious (was 3.50)
_INTERIOR_SEARCH_THRESHOLD = 2.00          # keep symmetric with root default

# Quiescence: how many additional plies of carpet-only search to do after the
# main search hits depth==0. 1 ply = only OUR immediate carpet extension is
# searched; the opponent's reply falls back to the heuristic. 2 plies over-
# assumed opponents always carpet when possible, which empirically cost games.
_QUIESCENCE_MAX_DEPTH = 1
# Minimum carpet value to qualify as "noisy" (a plain 2-length carpet is
# only 2 points; we skip those in quiescence since they're close to the
# heuristic estimate).
_QUIESCENCE_MIN_CARPET = 3


def prune_and_order_moves(board, belief, is_root=False, allow_search=True,
                          desperate=False, cautious=False):
    moves = board.get_valid_moves(enemy=False, exclude_search=not allow_search)
    scored = []

    from game.enums import loc_after_direction
    my_pos = board.player_worker.get_location()
    best_idx = int(np.argmax(belief))
    rx, ry = best_idx % BOARD_SIZE, best_idx // BOARD_SIZE

    for m in moves:
        if m.move_type == MoveType.CARPET:
            if m.roll_length < 2:
                continue
            priority = 1000 + CARPET_POINTS_TABLE[m.roll_length] * 10

        elif m.move_type == MoveType.SEARCH:
            ev = _search_ev_at(belief, m.search_loc)
            if is_root:
                if desperate:
                    threshold = _ROOT_SEARCH_THRESHOLD_DESPERATE
                elif cautious:
                    threshold = _ROOT_SEARCH_THRESHOLD_CAUTIOUS
                else:
                    threshold = _ROOT_SEARCH_THRESHOLD_DEFAULT
            else:
                threshold = _INTERIOR_SEARCH_THRESHOLD
            if ev <= threshold:
                continue
            priority = 500 + ev * 50

        elif m.move_type == MoveType.PRIME:
            extensions = 0
            for dxy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = my_pos[0] + dxy[0], my_pos[1] + dxy[1]
                if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                    if board.get_cell((nx, ny)) == Cell.PRIMED:
                        extensions += 1
            nxt = loc_after_direction(my_pos, m.direction)
            dest_extensions = 0
            if 0 <= nxt[0] < BOARD_SIZE and 0 <= nxt[1] < BOARD_SIZE:
                for dxy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = nxt[0] + dxy[0], nxt[1] + dxy[1]
                    if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                        if board.get_cell((nx, ny)) == Cell.PRIMED:
                            dest_extensions += 1
            priority = 200 + extensions * 30 + dest_extensions * 10

        elif m.move_type == MoveType.PLAIN:
            nxt = loc_after_direction(my_pos, m.direction)
            cur_d = abs(rx - my_pos[0]) + abs(ry - my_pos[1])
            new_d = abs(rx - nxt[0]) + abs(ry - nxt[1])
            priority = 100 + (cur_d - new_d) * 5
        else:
            priority = 50

        scored.append((priority, m))

    scored.sort(key=lambda x: -x[0])
    return [m for _, m in scored]


# =============================================================================
# The search engine
# =============================================================================
class ExpectiMinimax:
    def __init__(self, transition_matrix, max_depth: int = 6,
                 cfg: "HeuristicConfig" = None,
                 disable_search: bool = False):
        self.T_T = np.array(transition_matrix, dtype=np.float64).T
        self.max_depth = max_depth
        self.cfg = cfg if cfg is not None else _DEFAULT_CONFIG
        self.disable_search = disable_search
        self.tt = {}
        self.deadline = float("inf")
        self.nodes = 0

        self._killers = {}          # depth -> up to 2 move-keys
        self._history = {}          # move-key -> cumulative cutoff score
        self._desperate = False
        self._cautious = False
        self._suppress_search_override = False

        initial = np.zeros(64, dtype=np.float64)
        initial[0] = 1.0
        self._respawn_belief = np.linalg.matrix_power(self.T_T, 1000) @ initial

    # --- public API --------------------------------------------------------
    def search(self, board, belief, time_budget: float, desperate: bool = False,
               cautious: bool = False, suppress_search: bool = False):
        """Iterative-deepening alpha-beta. Returns (best_move, value, depth)."""
        self.tt.clear()
        self._killers.clear()
        self._history.clear()
        self.nodes = 0
        self.deadline = time.time() + max(0.05, time_budget)
        self._desperate = desperate
        self._cautious = cautious and not desperate
        # Per-call override of disable_search. Used by the agent to suppress
        # searches for a few turns after consecutive misses, when belief is
        # likely miscalibrated and the EV formula is overrating search.
        self._suppress_search_override = suppress_search

        best_move, best_value = None, NEG_INF
        reached_depth = 0
        pv_move = None

        # Aspiration window: for depth >= 3, try a narrow window around the
        # previous iteration's value before falling back to full bounds. A
        # successful narrow search is ~2x faster due to tighter alpha-beta
        # cutoffs; on fail-high/low we repeat with full bounds (small waste).
        # Window widens on each failure (20 -> 60 -> full) to avoid thrashing.
        ASP_WINDOW = 20.0
        for depth in range(1, self.max_depth + 1):
            if time.time() > self.deadline - 0.05:
                break
            try:
                if depth >= 3 and best_value is not None and abs(best_value) < 500:
                    # Narrow window around previous best value
                    lo = best_value - ASP_WINDOW
                    hi = best_value + ASP_WINDOW
                    val, mv = self._ab(board, depth, lo, hi, belief,
                                       pv_move, is_root=True)
                    # On fail-high or fail-low, re-search with widened or full window.
                    if val is not None and (val <= lo or val >= hi):
                        val, mv = self._ab(board, depth, NEG_INF, POS_INF,
                                           belief, pv_move, is_root=True)
                else:
                    val, mv = self._ab(board, depth, NEG_INF, POS_INF, belief,
                                       pv_move, is_root=True)
            except TimeoutSignal:
                break
            if mv is not None:
                best_move = mv
                best_value = val
                pv_move = mv
                reached_depth = depth
            if best_value > 500 or best_value < -500:
                break

        if best_move is None:
            moves = prune_and_order_moves(board, belief, is_root=True,
                                          allow_search=True,
                                          desperate=desperate,
                                          cautious=self._cautious)
            if not moves:
                moves = board.get_valid_moves(enemy=False, exclude_search=False)
            best_move = moves[0] if moves else Move.search((0, 0))

        return best_move, best_value, reached_depth

    # --- core recursion ----------------------------------------------------
    def _ab(self, board, depth, alpha, beta, belief, pv_move, is_root=False):
        if (self.nodes & 1023) == 0 and time.time() > self.deadline:
            raise TimeoutSignal()
        self.nodes += 1

        if board.is_game_over():
            return terminal_value(board), None
        if depth == 0:
            # Quiescence: extend past the horizon on big carpet swings so we
            # don't stop one move before the opponent empties a 5-piece chain
            # (or before we do). Bounded to 2 plies of carpet-only search.
            return self._quiescence(board, belief, alpha, beta,
                                    _QUIESCENCE_MAX_DEPTH), None
        if (board.player_worker.turns_left <= 0
                and board.opponent_worker.turns_left <= 0):
            return terminal_value(board), None

        tt_key = self._state_key(board)
        tt_entry = self.tt.get(tt_key)
        tt_move = None
        if tt_entry is not None:
            tt_val, tt_depth, tt_flag, tt_move = tt_entry
            if tt_depth >= depth and not is_root:
                if tt_flag == TT_EXACT:
                    return tt_val, tt_move
                elif tt_flag == TT_LOWER and tt_val >= beta:
                    return tt_val, tt_move
                elif tt_flag == TT_UPPER and tt_val <= alpha:
                    return tt_val, tt_move

        allow_search = not (self.disable_search or self._suppress_search_override)
        moves = prune_and_order_moves(
            board, belief, is_root=is_root, allow_search=allow_search,
            desperate=self._desperate, cautious=self._cautious,
        )
        if not moves:
            moves = [Move.search((x, y))
                     for x in range(BOARD_SIZE) for y in range(BOARD_SIZE)]

        # Order: PV > TT > killers > history > base priority
        ordered = []
        seen = set()

        def _mk_key(m):
            return (int(m.move_type),
                    -1 if m.direction is None else int(m.direction),
                    m.roll_length,
                    m.search_loc)

        priority_moves = []
        if pv_move is not None: priority_moves.append(pv_move)
        if tt_move is not None: priority_moves.append(tt_move)

        for preferred in priority_moves:
            k = _mk_key(preferred)
            if k in seen:
                continue
            for m in moves:
                if _mk_key(m) == k:
                    ordered.append(m)
                    seen.add(k)
                    break
        for kk in self._killers.get(depth, []):
            if kk in seen:
                continue
            for m in moves:
                if _mk_key(m) == kk:
                    ordered.append(m)
                    seen.add(kk)
                    break

        tail = [m for m in moves if _mk_key(m) not in seen]
        tail.sort(key=lambda m: -self._history.get(_mk_key(m), 0))
        ordered.extend(tail)

        best_val = NEG_INF
        best_move = None
        original_alpha = alpha
        move_index = 0

        for m in ordered:
            child = board.forecast_move(m, check_ok=False)
            if child is None:
                continue

            # Child belief: fold expected post-search information.
            search_ev = 0.0
            if m.move_type == MoveType.SEARCH:
                idx = m.search_loc[1] * BOARD_SIZE + m.search_loc[0]
                p_hit = float(belief[idx])
                search_ev = 6.0 * p_hit - 2.0

                missed = belief.copy()
                missed[idx] = 0.0
                s = missed.sum()
                if s > 1e-12:
                    missed /= s
                else:
                    missed = self._respawn_belief
                child_belief = p_hit * self._respawn_belief + (1.0 - p_hit) * missed
            else:
                child_belief = belief

            child_belief = self.T_T @ child_belief
            child.reverse_perspective()

            # PVS + LMR. First move full window; later moves null window
            # with a possible re-search.
            # Endgame precision: disable LMR in the last 3 turns. Carpets
            # are dense, swings are large, and a reduced late-move might
            # miss the decisive capture/denial. Worth the extra cycles.
            reduction = 0
            turns_left_min = min(board.player_worker.turns_left,
                                 board.opponent_worker.turns_left)
            if (move_index >= 3 and depth >= 3
                and m.move_type in (MoveType.PLAIN, MoveType.PRIME)
                and turns_left_min > 3):
                reduction = 1

            if move_index == 0:
                child_val, _ = self._ab(child, depth - 1, -beta, -alpha,
                                        child_belief, None)
                val = -child_val + search_ev
            else:
                child_val, _ = self._ab(child, depth - 1 - reduction,
                                        -alpha - 1, -alpha,
                                        child_belief, None)
                val = -child_val + search_ev
                if val > alpha and val < beta:
                    child_val, _ = self._ab(child, depth - 1,
                                            -beta, -alpha,
                                            child_belief, None)
                    val = -child_val + search_ev

            move_index += 1

            if val > best_val:
                best_val = val
                best_move = m
            if val > alpha:
                alpha = val
            if alpha >= beta:
                k = _mk_key(m)
                kl = self._killers.setdefault(depth, [])
                if k not in kl:
                    kl.insert(0, k)
                    if len(kl) > 2:
                        kl.pop()
                self._history[k] = self._history.get(k, 0) + (depth * depth)
                break

        if best_val <= original_alpha:
            flag = TT_UPPER
        elif best_val >= beta:
            flag = TT_LOWER
        else:
            flag = TT_EXACT
        self.tt[tt_key] = (best_val, depth, flag, best_move)

        return best_val, best_move

    def _quiescence(self, board, belief, alpha, beta, q_depth):
        """
        Carpet-only negamax search past the main horizon. Prevents the classic
        tactical blunder where we stop evaluating one move before the opponent
        carpets a long chain.

        Stand-pat: the heuristic is treated as a safe lower bound (the side to
        move is never FORCED to carpet -- they can always just accept the
        heuristic eval by making some quiet move instead). So we short-circuit
        if stand_pat already beats beta, and only search moves that could
        improve on it.
        """
        if (self.nodes & 1023) == 0 and time.time() > self.deadline:
            raise TimeoutSignal()
        self.nodes += 1

        if board.is_game_over():
            return terminal_value(board)
        if (board.player_worker.turns_left <= 0
                and board.opponent_worker.turns_left <= 0):
            return terminal_value(board)

        stand_pat = heuristic_eval(board, belief, self.cfg)
        if q_depth <= 0:
            return stand_pat
        if stand_pat >= beta:
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat

        # Only big carpet swings qualify as "noisy". Not including SEARCH
        # because search_ev is already baked into the heuristic.
        noisy = []
        for m in board.get_valid_moves(enemy=False, exclude_search=True):
            if (m.move_type == MoveType.CARPET
                    and m.roll_length >= _QUIESCENCE_MIN_CARPET):
                noisy.append(m)
        if not noisy:
            return stand_pat

        # Biggest carpets first for better pruning.
        noisy.sort(key=lambda m: -CARPET_POINTS_TABLE[m.roll_length])

        best_val = stand_pat
        for m in noisy:
            child = board.forecast_move(m, check_ok=False)
            if child is None:
                continue
            child_belief = self.T_T @ belief
            child.reverse_perspective()
            val = -self._quiescence(child, child_belief,
                                    -beta, -alpha, q_depth - 1)
            if val > best_val:
                best_val = val
            if val > alpha:
                alpha = val
            if alpha >= beta:
                break
        return best_val

    def _state_key(self, board):
        return (
            board._primed_mask,
            board._carpet_mask,
            board.player_worker.get_location(),
            board.opponent_worker.get_location(),
            board.player_worker.points - board.opponent_worker.points,
            board.player_worker.turns_left,
            board.opponent_worker.turns_left,
        )
