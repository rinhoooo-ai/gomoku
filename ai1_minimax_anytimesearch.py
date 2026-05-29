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
# PART 1 — EVALUATION
# ===========================================================================

def score_window(p_cnt: int, o_cnt: int, open_ends: int = 0) -> int:
    """
    Score a single 5-cell window for ONE player.

    open_ends = number of free cells just outside the window (0, 1, or 2).
        open_ends == 2 : fully open  → double value (most dangerous)
        open_ends == 1 : half open   → normal value
        open_ends == 0 : both ends blocked → heavily discounted
    """
    if o_cnt > 0 or p_cnt == 0:
        return 0
    base = SCORES[p_cnt]
    if open_ends == 2:
        return base * 2
    elif open_ends == 1:
        return base
    else:
        return base // 4


def evaluate(board: Board, player: int) -> int:
    """Heuristic evaluation of the full board for `player`."""
    opponent = 3 - player
    score    = 0

    for (dr, dc) in DIRECTIONS:
        for r in range(board.size):
            for c in range(board.size):
                er, ec = r + 4 * dr, c + 4 * dc
                if not (0 <= er < board.size and 0 <= ec < board.size):
                    continue

                p_cnt = o_cnt = 0
                for i in range(5):
                    cell = board.grid[r + i * dr, c + i * dc]
                    if cell == player:
                        p_cnt += 1
                    elif cell == opponent:
                        o_cnt += 1

                before_r, before_c = r - dr, c - dc
                after_r,  after_c  = r + 5 * dr, c + 5 * dc

                open_ends = 0
                if (0 <= before_r < board.size and 0 <= before_c < board.size
                        and board.grid[before_r, before_c] == 0):
                    open_ends += 1
                if (0 <= after_r < board.size and 0 <= after_c < board.size
                        and board.grid[after_r, after_c] == 0):
                    open_ends += 1

                score += score_window(p_cnt, o_cnt, open_ends)
                score -= score_window(o_cnt, p_cnt, open_ends)

    return score


# ===========================================================================
# PART 2 — MOVE ORDERING
# ===========================================================================

def _count_dir(board: Board, r: int, c: int,
               dr: int, dc: int, player: int) -> int:
    """Count consecutive `player` pieces through (r,c) in both directions."""
    count = 1
    for sign in (1, -1):
        nr, nc = r + sign * dr, c + sign * dc
        while 0 <= nr < board.size and 0 <= nc < board.size and board.grid[nr, nc] == player:
            count += 1
            nr += sign * dr
            nc += sign * dc
    return min(count, 5)


def quick_score(board: Board, r: int, c: int, player: int) -> int:
    """
    Fast move ordering score. Accounts for open vs blocked ends.

    Defense weights:
        5 → 2.0, 4 → 1.5, 3 → 1.2, 2 → 0.70, 1 → 0.50
    """
    DEFENSE_W = {5: 2.0, 4: 1.5, 3: 1.2, 2: 0.70, 1: 0.50}
    opponent  = 3 - player
    score     = 0

    for (dr, dc) in DIRECTIONS:
        board.grid[r, c] = player
        cnt_player = _count_dir(board, r, c, dr, dc, player)

        open_ends = 0
        for sign in (1, -1):
            nr, nc = r + sign * dr, c + sign * dc
            while 0 <= nr < board.size and 0 <= nc < board.size and board.grid[nr, nc] == player:
                nr += sign * dr
                nc += sign * dc
            if 0 <= nr < board.size and 0 <= nc < board.size and board.grid[nr, nc] == 0:
                open_ends += 1
        board.grid[r, c] = 0

        base = SCORES[cnt_player]
        if open_ends == 2:
            score += base * 2
        elif open_ends == 1:
            score += base
        else:
            score += base // 4

        board.grid[r, c] = opponent
        cnt_opponent = _count_dir(board, r, c, dr, dc, opponent)
        board.grid[r, c] = 0
        score += DEFENSE_W[cnt_opponent] * SCORES[cnt_opponent]

    return score


CANDIDATE_RADIUS = 2
MAX_CANDIDATES   = 8


def get_candidate_moves(board: Board) -> list:
    """Return empty cells within CANDIDATE_RADIUS of any occupied cell."""
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


def get_sorted_moves(board: Board, player: int,
                     depth_remaining: int = 99,
                     last_move: Optional[Tuple[int, int]] = None) -> list:
    """
    Return candidate moves sorted by quick_score (descending).
    Dynamic candidate cap based on depth_remaining:
        depth >= 3 → MAX_CANDIDATES (8)
        depth == 2 → 5
        depth == 1 → 3
    Proximity bonus applied if last_move is provided.
    """
    if depth_remaining <= 1:
        max_cand = 3
    elif depth_remaining <= 2:
        max_cand = 5
    else:
        max_cand = MAX_CANDIDATES

    moves = get_candidate_moves(board)

    def sort_key(move):
        qs = quick_score(board, move[0], move[1], player)
        if last_move is not None:
            dist = abs(move[0] - last_move[0]) + abs(move[1] - last_move[1])
            qs  += max(0, (5 - dist)) * 500
        return qs

    moves = sorted(moves, key=sort_key, reverse=True)
    return moves[:max_cand]


# ===========================================================================
# PART 2.5 — THREAT DETECTION
# ===========================================================================

def _find_threat_move(board: Board, player: int, min_count: int) -> Optional[Tuple[int, int]]:
    """
    Window-based threat detection — correctly handles gap patterns like X X _ X.
    Scans all 5-cell windows overlapping occupied cells.
    Returns the empty cell in the first window with (min_count - 1) player
    pieces and 0 opponent pieces.
    """
    opponent = 3 - player
    occupied = list(zip(*board.grid.nonzero()))
    if not occupied:
        return None

    seen = set()
    for (pr, pc) in occupied:
        for (dr, dc) in DIRECTIONS:
            for offset in range(-4, 1):
                sr = pr + offset * dr
                sc = pc + offset * dc
                er = sr + 4 * dr
                ec = sc + 4 * dc
                if not (0 <= sr < board.size and 0 <= sc < board.size and
                        0 <= er < board.size and 0 <= ec < board.size):
                    continue
                key = (sr, sc, dr, dc)
                if key in seen:
                    continue
                seen.add(key)

                p_cnt = o_cnt = 0
                for i in range(5):
                    cell = board.grid[sr + i * dr, sc + i * dc]
                    if cell == player:
                        p_cnt += 1
                    elif cell == opponent:
                        o_cnt += 1

                if o_cnt == 0 and p_cnt == min_count - 1:
                    for i in range(5):
                        nr, nc = sr + i * dr, sc + i * dc
                        if board.grid[nr, nc] == 0:
                            return (nr, nc)

    return None


def _find_open3_block(board: Board, player: int) -> Optional[Tuple[int, int]]:
    """
    Collect all opponent open-3 threats and return the best block move.

    Threats are bucketed by openness:
        both_open (2 free ends) — most dangerous, prioritized first
        one_open  (1 free end)  — less urgent
        fully blocked (0 ends)  — ignored, not a real threat

    Within each bucket, the candidate with highest quick_score is returned.
    This ensures edge-blocked three-in-rows (only 1 open end) are never
    confused with true open-3 threats (both ends free).
    """
    opponent = 3 - player
    occupied = list(zip(*board.grid.nonzero()))
    if not occupied:
        return None

    seen = set()
    both_open_moves = []   # (quick_score, move)
    one_open_moves  = []

    for (pr, pc) in occupied:
        for (dr, dc) in DIRECTIONS:
            for offset in range(-4, 1):
                sr, sc = pr + offset * dr, pc + offset * dc
                er, ec = sr + 4 * dr,      sc + 4 * dc
                if not (0 <= sr < board.size and 0 <= sc < board.size and
                        0 <= er < board.size and 0 <= ec < board.size):
                    continue
                key = (sr, sc, dr, dc)
                if key in seen:
                    continue
                seen.add(key)

                o_cnt = p_cnt = 0
                for i in range(5):
                    cell = board.grid[sr + i * dr, sc + i * dc]
                    if cell == opponent:
                        o_cnt += 1
                    elif cell == player:
                        p_cnt += 1

                # Need exactly 3 opponent pieces, none of ours blocking
                if o_cnt != 3 or p_cnt != 0:
                    continue

                before_r, before_c = sr - dr,     sc - dc
                after_r,  after_c  = sr + 5 * dr, sc + 5 * dc

                open_before = (0 <= before_r < board.size and
                               0 <= before_c < board.size and
                               board.grid[before_r, before_c] == 0)
                open_after  = (0 <= after_r < board.size and
                               0 <= after_c < board.size and
                               board.grid[after_r, after_c] == 0)

                open_ends = int(open_before) + int(open_after)
                if open_ends == 0:
                    continue   # fully blocked — not a real threat

                # Collect all blockable cells: open ends + empty gaps inside window
                candidates = []
                if open_before:
                    candidates.append((int(before_r), int(before_c)))
                if open_after:
                    candidates.append((int(after_r), int(after_c)))
                for i in range(5):
                    nr, nc = sr + i * dr, sc + i * dc
                    if board.grid[nr, nc] == 0:
                        candidates.append((int(nr), int(nc)))
                if not candidates:
                    continue

                best = max(candidates,
                           key=lambda m: quick_score(board, m[0], m[1], player))
                qs   = quick_score(board, best[0], best[1], player)

                if open_ends == 2:
                    both_open_moves.append((qs, best))
                else:
                    one_open_moves.append((qs, best))

    # Return the best move from the most dangerous bucket
    if both_open_moves:
        return max(both_open_moves, key=lambda x: x[0])[1]
    if one_open_moves:
        return max(one_open_moves, key=lambda x: x[0])[1]
    return None


# ===========================================================================
# PART 3 — MINIMAX WITH ALPHA-BETA
# ===========================================================================

def _minimax(board: Board, depth: int, alpha: float, beta: float,
             is_maximizing: bool, player: int,
             last_move: Optional[Tuple[int, int]],
             start: float, time_limit: float) -> Optional[int]:
    """
    Minimax with Alpha-Beta Pruning + time guard + dynamic candidate reduction.
    Returns None when time budget is exhausted.
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

    sorted_moves = get_sorted_moves(
        board,
        player if is_maximizing else opponent,
        depth_remaining=depth
    )

    if is_maximizing:
        best = -math.inf
        for (r, c) in sorted_moves:
            board.make_move(r, c, player)
            score = _minimax(board, depth - 1, alpha, beta, False, player,
                             (r, c), start, time_limit)
            board.undo_move(r, c)
            if score is None:
                return None
            best  = max(best, score)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best
    else:
        best = math.inf
        for (r, c) in sorted_moves:
            board.make_move(r, c, opponent)
            score = _minimax(board, depth - 1, alpha, beta, True, player,
                             (r, c), start, time_limit)
            board.undo_move(r, c)
            if score is None:
                return None
            best = min(best, score)
            beta = min(beta, best)
            if beta <= alpha:
                break
        return best


# ===========================================================================
# PART 4 — ANYTIME SEARCH
# ===========================================================================

def get_best_move(board: Board, player: int,
                  time_limit: float = 15.0,
                  last_move: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
    """
    Anytime search via Iterative Deepening + Alpha-Beta.

    Pre-checks (priority order):
        1. Immediate win
        2. Block opponent immediate win
        3. Own open-4 (push to 5 next turn)
        4. Block opponent open-4
        5. Block opponent open-3 (both-open threats first, then one-open)
    """
    MAX_DEPTH = 20
    start     = time.time()
    opponent  = 3 - player

    # PRE-CHECK 1: Immediate win
    move = _find_threat_move(board, player, 5)
    if move:
        return move

    # PRE-CHECK 2: Block opponent immediate win
    move = _find_threat_move(board, opponent, 5)
    if move:
        return move

    # PRE-CHECK 3: Own open-4
    move = _find_threat_move(board, player, 4)
    if move:
        return move

    # PRE-CHECK 4: Block opponent open-4 — opponent is 1 move from winning.
    # Use check_win directly instead of _find_threat_move to avoid false
    # positives where a 3-in-row window is mistaken for an open-4 threat.
    for (r, c) in get_candidate_moves(board):
        board.grid[r, c] = opponent
        if board.check_win(r, c, opponent):
            board.grid[r, c] = 0
            return (int(r), int(c))
        board.grid[r, c] = 0

    # PRE-CHECK 5: Block opponent open-3 (prioritizes both-open over one-open)
    move = _find_open3_block(board, player)
    if move:
        return move

    # MINIMAX — iterative deepening
    best_move  = None
    best_depth = 0

    fallback = get_sorted_moves(board, player, last_move=last_move)
    if fallback:
        best_move = fallback[0]

    for depth in range(1, MAX_DEPTH + 1):
        if time.time() - start >= time_limit:
            break

        candidate_move  = None
        candidate_score = -math.inf
        sorted_moves    = get_sorted_moves(board, player,
                                           depth_remaining=depth,
                                           last_move=last_move)
        timed_out = False

        for (r, c) in sorted_moves:
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

    # 1. score_window
    assert score_window(3, 0, 2) == SCORES[3] * 2
    assert score_window(3, 0, 1) == SCORES[3]
    assert score_window(3, 0, 0) == SCORES[3] // 4
    assert score_window(3, 1, 2) == 0
    assert score_window(0, 0, 2) == 0
    print("PASS: score_window")

    # 2. evaluate empty
    assert evaluate(board, 1) == 0
    assert evaluate(board, 2) == 0
    print("PASS: evaluate empty board")

    # 3. evaluate gap pattern
    board.make_move(0, 0, 1); board.make_move(0, 1, 1)
    board.make_move(0, 3, 1); board.make_move(0, 4, 1)
    assert evaluate(board, 1) >= SCORES[4]
    board.grid[:] = 0
    print("PASS: evaluate gap pattern")

    # 4. open-3 scores higher than blocked-3
    board.make_move(7, 5, 1); board.make_move(7, 6, 1); board.make_move(7, 7, 1)
    score_open = evaluate(board, 1)
    board.make_move(7, 4, 2)
    score_blocked = evaluate(board, 1)
    assert score_open > score_blocked
    board.grid[:] = 0
    print("PASS: evaluate open > blocked")

    # 5. evaluate symmetry
    board.make_move(0, 0, 1); board.make_move(1, 0, 1)
    board.make_move(2, 0, 2); board.make_move(3, 0, 2)
    assert evaluate(board, 1) == -evaluate(board, 2)
    board.grid[:] = 0
    print("PASS: evaluate symmetry")

    # 6. Block opponent open-4
    for col in range(4):
        board.make_move(0, col, 2)
    move = get_best_move(board, 1, time_limit=15.0)
    assert move[0] == 0, f"should block row 0, got {move}"
    board.grid[:] = 0
    print("PASS: blocks opponent open-4")

    # 7. Take winning move
    for col in range(4):
        board.make_move(0, col, 1)
    move = get_best_move(board, 1, time_limit=15.0)
    assert move == (0, 4), f"should complete 5-in-row, got {move}"
    board.grid[:] = 0
    print("PASS: takes win")

    # 8. Block opponent open-3 (both ends open)
    board.make_move(7, 7, 2); board.make_move(7, 8, 2); board.make_move(7, 9, 2)
    move = get_best_move(board, 1, time_limit=15.0)
    assert move in ((7, 6), (7, 10)), f"should block open-3, got {move}"
    board.grid[:] = 0
    print("PASS: blocks opponent open-3")

    # 9. Ignore one-open-end 3-in-row, block the real open-3 instead
    # Setup: opponent has blocked-end 3 at row 0 (edge blocks one end)
    #        AND a fresh open-3 at row 7
    board.make_move(0, 0, 2); board.make_move(0, 1, 2); board.make_move(0, 2, 2)  # edge-blocked
    board.make_move(7, 7, 2); board.make_move(7, 8, 2); board.make_move(7, 9, 2)  # open-3
    move = get_best_move(board, 1, time_limit=15.0)
    assert move in ((7, 6), (7, 10)), \
        f"should block open-3 at row 7, not edge-3 at row 0, got {move}"
    board.grid[:] = 0
    print("PASS: prioritizes open-3 over edge-blocked 3")

    # 10. Gap pattern block: X _ X X
    board.make_move(5, 5, 2); board.make_move(5, 7, 2); board.make_move(5, 8, 2)
    move = get_best_move(board, 1, time_limit=15.0)
    assert move in ((5, 4), (5, 6), (5, 9)), \
        f"should block gap threat, got {move}"
    board.grid[:] = 0
    print("PASS: blocks gap pattern")

    # 11. Anytime within time limit
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
    assert elapsed <= 16.0, f"exceeded time: {elapsed:.1f}s"
    board.grid[:] = 0
    print(f"PASS: anytime returned in {elapsed:.2f}s")

    print("\n=== All tests passed! ===")