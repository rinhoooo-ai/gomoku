from game import Board
from typing import Tuple, Optional
import math
import time

# ---------------------------------------------------------------------------
# SCORING CONSTANTS
# ---------------------------------------------------------------------------
SCORES = {1: 10, 2: 100, 3: 1_000, 4: 10_000, 5: 1_000_000}
WIN_SCORE  =  SCORES[5]
LOSE_SCORE = -SCORES[5]

DIRECTIONS = [
    (0, 1),   # horizontal
    (1, 0),   # vertical
    (1, 1),   # diagonal
    (1, -1),  # anti-diagonal
]

# ===========================================================================
# PART 1 — EVALUATION (leaf node, no more search after this)
# ===========================================================================

def score_window(p_cnt: int, o_cnt: int) -> int:
    """Score a single 5-cell window for ONE player."""
    if p_cnt > 0 and o_cnt == 0:
        return SCORES[p_cnt]
    return 0


def evaluate(board: Board, player: int) -> int:
    """Heuristic evaluation of the full board for `player` at a leaf node."""
    opponent = 3 - player
    score    = 0

    for (dr, dc) in DIRECTIONS:
        for r in range(board.size):
            for c in range(board.size):
                er, ec = r + 4 * dr, c + 4 * dc
                if not (0 <= er < board.size and 0 <= ec < board.size):
                    continue

                p_cnt = 0
                o_cnt = 0

                for i in range(5):
                    if board.grid[r + i * dr, c + i * dc] == player:
                        p_cnt += 1
                    elif board.grid[r + i * dr, c + i * dc] == opponent:
                        o_cnt += 1

                score += score_window(p_cnt, o_cnt)
                score -= score_window(o_cnt, p_cnt)

    return score


# ===========================================================================
# PART 2 — MOVE ORDERING  (approximate score, fast)
# ===========================================================================

def _count_dir(board: Board, r: int, c: int,
               dr: int, dc: int, player: int) -> int:
    """
    Count consecutive `player` pieces through (r, c) in direction (dr, dc),
    counting both ways (sign = +1 and sign = -1).
    """
    count = 1  # the piece at (r, c) itself
    for sign in (1, -1):
        nr, nc = r + sign * dr, c + sign * dc
        while 0 <= nr < board.size and 0 <= nc < board.size and board.grid[nr, nc] == player:
            count += 1
            nr += sign * dr
            nc += sign * dc
    return min(count, 5)


def quick_score(board: Board, r: int, c: int, player: int) -> int:
    """
    Fast PROMISING score for placing `player` at empty cell (r, c).
    Used only for MOVE ORDERING — does not need to be perfect.

    Method:
        For each of the 4 DIRECTIONS:
            1. Tentatively place `player`   -> count consecutive → add SCORES[cnt]
            2. Tentatively place `opponent` -> count consecutive → add SCORES[cnt] * defense_weight
        Restore cell to 0 after each tentative placement.

    Defense weights (opponent consecutive count -> weight):
        5 → 2.0  (opponent wins → must block at all costs)
        4 → 1.5  (opponent 1-away → block strongly)
        3 → 1.2  (opponent open-3 → blocking scores higher than attacking)
        2 → 0.70
        1 → 0.50

    INPUT:
        board   — Board  (cell r,c must be empty: board.grid[r,c] == 0)
        r, c    — candidate move position
        player  — 1 or 2
    OUTPUT:
        int — higher = more promising move
    """
    DEFENSE_W = {5: 2.0, 4: 1.5, 3: 1.2, 2: 0.70, 1: 0.50}
    opponent  = 3 - player
    score     = 0

    for (dr, dc) in DIRECTIONS:
        # Tentatively place `player`, count consecutive, restore.
        board.grid[r, c] = player
        cnt_player = _count_dir(board, r, c, dr, dc, player)
        board.grid[r, c] = 0

        # Tentatively place `opponent`, count consecutive, restore.
        board.grid[r, c] = opponent
        cnt_opponent = _count_dir(board, r, c, dr, dc, opponent)
        board.grid[r, c] = 0

        score += SCORES[cnt_player]
        score += DEFENSE_W[cnt_opponent] * SCORES[cnt_opponent]

    return score


CANDIDATE_RADIUS = 2


def get_candidate_moves(board: Board) -> list:
    """
    Return empty cells within CANDIDATE_RADIUS of any occupied cell.

    Restricts the search space from all ~225 empty cells on a 15x15 board
    to the ~20-40 cells that are actually relevant, enabling much deeper
    iterative-deepening search within the time budget.

    If the board is empty, return the center cell as the only candidate
    (opening move convention for Gomoku).
    """
    occupied = list(zip(*board.grid.nonzero()))
    if not occupied:
        center = board.size // 2
        return [(center, center)]

    candidates = set()
    for (r, c) in occupied:
        for dr in range(-CANDIDATE_RADIUS, CANDIDATE_RADIUS + 1):
            for dc in range(-CANDIDATE_RADIUS, CANDIDATE_RADIUS + 1):
                nr, nc = r + dr, c + dc
                if (0 <= nr < board.size and
                        0 <= nc < board.size and
                        board.grid[nr, nc] == 0):
                    candidates.add((nr, nc))

    return list(candidates)


MAX_CANDIDATES = 10


def get_sorted_moves(board: Board, player: int,
                     last_move: Optional[Tuple[int, int]] = None) -> list:
    """
    Return candidate cells sorted by quick_score (descending).
    Candidates are restricted to CANDIDATE_RADIUS of occupied cells.
    Better moves first -> alpha-beta prunes more aggressively.

    If last_move is provided (the opponent's most recent move), cells closer
    to it receive a proximity bonus — forcing the search to examine direct
    responses first before exploring unrelated parts of the board.
    """
    moves = get_candidate_moves(board)

    def sort_key(move):
        qs = quick_score(board, move[0], move[1], player)
        if last_move is not None:
            dist = abs(move[0] - last_move[0]) + abs(move[1] - last_move[1])
            qs  += max(0, (5 - dist)) * 500   # proximity bonus decays with distance
        return qs

    moves = sorted(moves, key=sort_key, reverse=True)
    return moves[:MAX_CANDIDATES]


# ===========================================================================
# PART 2.5 — EXPLICIT THREAT CHECKS  (O(candidates), runs before minimax)
# ===========================================================================

def _is_open_end(board: Board, r: int, c: int, dr: int, dc: int) -> bool:
    """
    Return True if cell (r, c) in direction (dr, dc) is an open end —
    i.e. it is within bounds AND empty (not a wall, not an opponent stone).
    Used to distinguish true open threats from edge-blocked or capped runs.
    """
    nr, nc = r + dr, c + dc
    return (0 <= nr < board.size and
            0 <= nc < board.size and
            board.grid[nr, nc] == 0)


def _find_threat_move(board: Board, player: int, min_count: int,
                      require_open: bool = False) -> Optional[Tuple[int, int]]:
    """
    Scan all candidates: return the first empty cell where placing `player`
    creates a consecutive run of >= min_count in any direction.

    If require_open=True, only count threats where at least one end of the
    run is open (not blocked by board edge or opponent stone).  This prevents
    treating edge-hugging 3-in-a-row as a genuine open-3 threat — a run
    blocked by the board boundary on one side is significantly weaker and
    should not trigger a forced pre-check response.

    Used to detect:
        min_count=5 → immediate win          (require_open=False — win is win)
        min_count=4 → open-4 / 4-in-a-row   (require_open=False — still urgent)
        min_count=3 → open-3 threat          (require_open=True  — edge-3 is weak)

    INPUT:
        board        : Board
        player       : the player to check for
        min_count    : minimum consecutive count to qualify as a threat
        require_open : if True, skip runs where both ends are blocked/off-board
    OUTPUT:
        (r, c) of the threatening cell, or None if no threat found
    """
    opponent = 3 - player
    for (r, c) in get_candidate_moves(board):
        board.grid[r, c] = player
        for (dr, dc) in DIRECTIONS:
            cnt = _count_dir(board, r, c, dr, dc, player)
            if cnt >= min_count:
                if require_open:
                    # Walk to the far end of the run in each direction,
                    # then check if that end is open.
                    open_ends = 0
                    for sign in (1, -1):
                        # Step past the run in this sign direction
                        nr, nc = r + sign * dr, c + sign * dc
                        while (0 <= nr < board.size and
                               0 <= nc < board.size and
                               board.grid[nr, nc] == player):
                            nr += sign * dr
                            nc += sign * dc
                        # Now (nr, nc) is just beyond the run — check if open
                        if (0 <= nr < board.size and
                                0 <= nc < board.size and
                                board.grid[nr, nc] == 0):
                            open_ends += 1
                    if open_ends == 0:
                        continue   # both ends blocked — not a real open threat
                board.grid[r, c] = 0
                return (r, c)
        board.grid[r, c] = 0
    return None


# ===========================================================================
# PART 3 — MINIMAX WITH ALPHA-BETA  (depth-limited, timed)
# ===========================================================================

def _minimax(board: Board, depth: int, alpha: float, beta: float,
             is_maximizing: bool, player: int,
             last_move: Optional[Tuple[int, int]],
             start: float, time_limit: float) -> Optional[int]:
    """
    Minimax with Alpha-Beta Pruning + time guard.

    Returns None when the wall-clock budget is exhausted (caller discards
    this incomplete result and keeps the best result from the previous depth).

    TERMINAL CONDITIONS  (check in this order):
        1. time.time() - start >= time_limit  -> return None
        2. last_move caused a win             -> return WIN_SCORE or LOSE_SCORE
        3. depth == 0                         -> return evaluate(board, player)
        4. board.is_full()                    -> return 0  (draw)

    RECURSIVE STEP:
        sorted_moves = get_sorted_moves(board, player  if is_maximizing
                                                       else opponent)
        Maximizer: tries all moves for `player`,   keeps best, updates alpha.
        Minimizer: tries all moves for `opponent`, keeps best, updates beta.
        Alpha-Beta cut: if beta <= alpha → break.
        If any recursive call returns None → propagate None immediately.

    INPUT:
        board         : Board
        depth         : remaining search depth
        alpha, beta   : pruning bounds  (init: -inf, +inf)
        is_maximizing : True when it is `player`'s turn
        player        : the MAX player (fixed throughout the tree)
        last_move     : (r, c) of the move just played, or None at root
        start         : time.time() at search start
        time_limit    : seconds budget
    OUTPUT:
        int  : heuristic value of the position  (from `player`'s perspective)
        None : time expired mid-search
    """
    opponent = 3 - player

    if time.time() - start >= time_limit:
        return None

    if last_move is not None:
        last_player = opponent if is_maximizing else player
        if board.check_win(last_move[0], last_move[1], last_player):
            return WIN_SCORE if last_player == player else LOSE_SCORE

    if depth == 0:
        return evaluate(board, player)

    if board.is_full():
        return 0

    sorted_moves = get_sorted_moves(board, player if is_maximizing else opponent)

    if is_maximizing:
        best = -math.inf
        for (r, c) in sorted_moves:
            board.make_move(r, c, player)
            score = _minimax(board, depth - 1, alpha, beta, False, player, (r, c), start, time_limit)
            board.undo_move(r, c)

            if score is None:
                return None
            best = max(best, score)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best

    else:
        best = math.inf
        for (r, c) in sorted_moves:
            board.make_move(r, c, opponent)
            score = _minimax(board, depth - 1, alpha, beta, True, player, (r, c), start, time_limit)
            board.undo_move(r, c)

            if score is None:
                return None
            best = min(best, score)
            beta = min(beta, best)
            if beta <= alpha:
                break
        return best


# ===========================================================================
# PART 4 — ANYTIME SEARCH  (iterative deepening)
# ===========================================================================

def get_best_move(board: Board, player: int,
                  time_limit: float = 15.0,
                  last_move: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
    """
    Anytime search via Iterative Deepening + Alpha-Beta.

    last_move — the opponent's most recent move (r, c), or None.
    When provided, get_sorted_moves biases candidates toward cells near
    last_move so the search examines direct responses first.

    Pre-checks (in priority order) before running minimax:
        1. Immediate win  — take it instantly.
        2. Immediate block (opponent 1-away) — block instantly.
        3. Own open-4    — push to 5, win next turn regardless.
        4. Opponent fork — block cell that would create 2+ threats.
        5. Opponent open-3 — let minimax pick best block among endpoints.

    Then falls through to full iterative-deepening minimax.

    CONSTANTS you may use:
        MAX_DEPTH = 20
    """
    MAX_DEPTH  = 20
    start      = time.time()
    opponent   = 3 - player

    # ------------------------------------------------------------------
    # PRE-CHECK 1 — Immediate win
    # ------------------------------------------------------------------
    move = _find_threat_move(board, player, 5)
    if move:
        return move

    # ------------------------------------------------------------------
    # PRE-CHECK 2 — Immediate block (opponent wins next move)
    # ------------------------------------------------------------------
    move = _find_threat_move(board, opponent, 5)
    if move:
        return move

    # ------------------------------------------------------------------
    # PRE-CHECK 3 — Own open-4: push to 5 next turn
    # ------------------------------------------------------------------
    move = _find_threat_move(board, player, 4)
    if move:
        return move

    # ------------------------------------------------------------------
    # PRE-CHECK 4 — Opponent fork: one opponent move creates 2+ threats.
    # If opponent can play a single cell that simultaneously builds 2
    # directions to count >= 4, block that cell before the fork lands.
    # ------------------------------------------------------------------
    for (r, c) in get_candidate_moves(board):
        board.grid[r, c] = opponent
        threat_count = sum(
            1 for (dr, dc) in DIRECTIONS
            if _count_dir(board, r, c, dr, dc, opponent) >= 4
        )
        board.grid[r, c] = 0
        if threat_count >= 2:
            return (r, c)

    # ------------------------------------------------------------------
    # PRE-CHECK 5 — Opponent open-3.
    # Collect all cells where opponent placing creates an open-3.
    # Edge-blocked runs (both ends closed) are excluded — weak threat.
    # ------------------------------------------------------------------
    threat_cells = []
    for (r, c) in get_candidate_moves(board):
        board.grid[r, c] = opponent
        for (dr, dc) in DIRECTIONS:
            if _count_dir(board, r, c, dr, dc, opponent) >= 3:
                open_ends = 0
                for sign in (1, -1):
                    nr, nc = r + sign * dr, c + sign * dc
                    while (0 <= nr < board.size and
                           0 <= nc < board.size and
                           board.grid[nr, nc] == opponent):
                        nr += sign * dr
                        nc += sign * dc
                    if (0 <= nr < board.size and
                            0 <= nc < board.size and
                            board.grid[nr, nc] == 0):
                        open_ends += 1
                if open_ends > 0:
                    threat_cells.append((r, c))
                    break
        board.grid[r, c] = 0

    # If opponent has open-3 BUT we also have open-3, do NOT force block —
    # let minimax decide: scaling our own threat to open-4 forces opponent
    # to defend, which is often stronger than reacting to their open-3.
    # Only force-block when we have no counter-threat of our own.
    my_open3 = _find_threat_move(board, player, 3, require_open=True)
    if threat_cells and not my_open3:
        best_block = max(threat_cells,
                         key=lambda m: quick_score(board, m[0], m[1], player))
        return best_block

    # ------------------------------------------------------------------
    # MINIMAX — iterative deepening
    # ------------------------------------------------------------------
    best_move  = None
    best_depth = 0

    # Fallback: pick first legal move so we never return None
    fallback = get_sorted_moves(board, player, last_move)
    if fallback:
        best_move = fallback[0]

    # Build root candidates: always include our own open-3 scaling moves
    # at the top so minimax evaluates them first, then fill with sorted moves.
    root_seed = []
    if my_open3:
        root_seed.append(my_open3)
    root_seed += [m for m in get_sorted_moves(board, player, last_move)
                  if m not in root_seed]
    root_seed = root_seed[:MAX_CANDIDATES]

    for depth in range(1, MAX_DEPTH + 1):
        if time.time() - start >= time_limit:
            break

        candidate_move  = None
        candidate_score = -math.inf

        root_moves = root_seed

        timed_out = False

        for (r, c) in root_moves:
            if time.time() - start >= time_limit:
                timed_out = True
                break

            board.make_move(r, c, player)
            score = _minimax(board, depth - 1, -math.inf, math.inf,
                             False, player, (r, c), start, time_limit)
            board.undo_move(r, c)

            if score is None:
                timed_out = True
                break

            if score > candidate_score:
                candidate_score = score
                candidate_move  = (r, c)

            if candidate_score >= WIN_SCORE:
                best_move  = candidate_move
                best_depth = depth
                timed_out  = False
                break

        if not timed_out and candidate_move is not None:
            best_move  = candidate_move
            best_depth = depth

        if timed_out or candidate_score >= WIN_SCORE:
            break

    print(f"[Minimax] depth reached: {best_depth} | time: {time.time()-start:.2f}s")
    return best_move


# ===========================================================================
# TESTS
# ===========================================================================
if __name__ == "__main__":
    board = Board(15)

    assert score_window(3, 0) == SCORES[3]
    assert score_window(3, 1) == 0
    assert score_window(0, 0) == 0
    assert score_window(5, 0) == SCORES[5]
    print("PASS: score_window")

    assert evaluate(board, 1) == 0
    assert evaluate(board, 2) == 0
    print("PASS: evaluate empty board")

    board.make_move(0, 0, 1); board.make_move(0, 1, 1)
    board.make_move(0, 3, 1); board.make_move(0, 4, 1)
    assert evaluate(board, 1) >= SCORES[4]
    board.grid[:] = 0
    print("PASS: evaluate gap pattern")

    board.make_move(0, 0, 1); board.make_move(1, 0, 1)
    board.make_move(2, 0, 2); board.make_move(3, 0, 2)
    assert evaluate(board, 1) == -evaluate(board, 2)
    board.grid[:] = 0
    print("PASS: evaluate symmetry")

    for col in range(4):
        board.make_move(0, col, 2)
    move = get_best_move(board, 1, time_limit=15.0)
    assert move[0] == 0
    board.grid[:] = 0
    print("PASS: get_best_move blocks opponent 4")

    for col in range(4):
        board.make_move(0, col, 1)
    move = get_best_move(board, 1, time_limit=15.0)
    assert move == (0, 4)
    board.grid[:] = 0
    print("PASS: get_best_move takes win")

    # Block open-3: opponent has 3-in-row, AI must block one end
    board.make_move(7, 7, 2); board.make_move(7, 8, 2); board.make_move(7, 9, 2)
    move = get_best_move(board, 1, time_limit=15.0)
    assert move in ((7, 6), (7, 10)), f"AI should block open-3, got {move}"
    board.grid[:] = 0
    print("PASS: get_best_move blocks opponent open-3")

    import random
    random.seed(42)
    for _ in range(10):
        r, c = random.randint(5, 9), random.randint(5, 9)
        if board.grid[r, c] == 0:
            board.make_move(r, c, random.choice([1, 2]))
    t0      = time.time()
    move    = get_best_move(board, 1, time_limit=15.0)
    elapsed = time.time() - t0
    assert move is not None
    assert elapsed <= 16.0
    board.grid[:] = 0
    print(f"PASS: anytime search returned in {elapsed:.2f}s")

    print("\n=== All tests passed! ===")