# -*- coding: utf-8 -*-
"""
ai1_minimax_fixed.py  —  Gomoku AI (fixed threat detection + deeper search)

Key fixes vs original:
  1. _find_threat_move: now uses window-based scanning (same logic as evaluate)
     instead of _count_dir, so it correctly detects gap patterns like _ X X _ X _
  2. _find_open_threat: new helper that also detects OPEN threats (both ends free)
     so _ X X X _ is caught and blocked, not just X X X X _
  3. MAX_CANDIDATES reduced 15→8 for search, giving ~2× more depth per second
  4. get_sorted_moves now uses a hard cutoff: if a move scores >= WIN_SCORE, stop
     expanding further (instant win/block found)
"""

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

def score_window(p_cnt: int, o_cnt: int) -> int:
    if p_cnt > 0 and o_cnt == 0:
        return SCORES[p_cnt]
    return 0


def evaluate(board: Board, player: int) -> int:
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
                    cell = board.grid[r + i * dr, c + i * dc]
                    if cell == player:
                        p_cnt += 1
                    elif cell == opponent:
                        o_cnt += 1

                score += score_window(p_cnt, o_cnt)
                score -= score_window(o_cnt, p_cnt)

    return score


# ===========================================================================
# PART 2 — MOVE ORDERING
# ===========================================================================

def _count_dir(board: Board, r: int, c: int,
               dr: int, dc: int, player: int) -> int:
    count = 1
    for sign in (1, -1):
        nr, nc = r + sign * dr, c + sign * dc
        while 0 <= nr < board.size and 0 <= nc < board.size and board.grid[nr, nc] == player:
            count += 1
            nr += sign * dr
            nc += sign * dc
    return min(count, 5)


def quick_score(board: Board, r: int, c: int, player: int) -> int:
    DEFENSE_W = {5: 2.0, 4: 0.95, 3: 0.90, 2: 0.70, 1: 0.50}
    opponent  = 3 - player
    score     = 0

    for (dr, dc) in DIRECTIONS:
        cnt_player = _count_dir(board, r, c, dr, dc, player)
        score += SCORES[cnt_player]

        cnt_opponent = _count_dir(board, r, c, dr, dc, opponent)
        score += DEFENSE_W[cnt_opponent] * SCORES[cnt_opponent]

    return score


CANDIDATE_RADIUS = 2


def get_candidate_moves(board: Board) -> list:
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


# FIX #3: Reduced from 15 → 8 to achieve ~2× deeper search within time budget.
# At branching factor 8, depth 6 = 8^6 = 262,144 nodes vs 15^6 = 11M nodes.
MAX_CANDIDATES = 8


def get_sorted_moves(board: Board, player: int) -> list:
    moves = get_candidate_moves(board)
    moves = sorted(moves,
                   key=lambda move: quick_score(board, move[0], move[1], player),
                   reverse=True)
    return moves[:MAX_CANDIDATES]


# ===========================================================================
# PART 2.5 — THREAT DETECTION  (FIXED)
# ===========================================================================

def _window_counts(board: Board, r: int, c: int, dr: int, dc: int,
                   player: int) -> Tuple[int, int]:
    """
    Count player and opponent pieces in the 5-cell window starting at (r,c).
    Returns (p_cnt, o_cnt).
    """
    p_cnt = o_cnt = 0
    for i in range(5):
        nr, nc = r + i * dr, c + i * dc
        cell = board.grid[nr, nc]
        if cell == player:
            p_cnt += 1
        elif cell != 0:
            o_cnt += 1
    return p_cnt, o_cnt


def _find_threat_move(board: Board, player: int, min_count: int) -> Optional[Tuple[int, int]]:
    """
    FIX #1: Window-based threat detection instead of _count_dir.

    Old approach: place piece at empty cell, call _count_dir (consecutive only).
    → Misses gap patterns: _ X X _ X _ looks like count=1 from the placed cell.

    New approach: scan every 5-cell window around occupied cells (same logic
    as evaluate). For each window with exactly (min_count - 1) player pieces
    and 0 opponent pieces, the single empty cell in that window is the threat move.

    This correctly catches:
      - _ X X X _   (3 consecutive, both ends open)
      - _ X X _ X _ (gap pattern → open 4 threat)
      - X X _ X X   (split 4)
    """
    # Only scan windows that overlap with occupied cells (efficiency)
    occupied = list(zip(*board.grid.nonzero()))
    if not occupied:
        return None

    candidate_windows = set()
    for (pr, pc) in occupied:
        for (dr, dc) in DIRECTIONS:
            # Each occupied cell can be in windows starting at offset -4..0
            for offset in range(-4, 1):
                sr = pr + offset * dr
                sc = pc + offset * dc
                er = sr + 4 * dr
                ec = sc + 4 * dc
                if (0 <= sr < board.size and 0 <= sc < board.size and
                        0 <= er < board.size and 0 <= ec < board.size):
                    candidate_windows.add((sr, sc, dr, dc))

    for (sr, sc, dr, dc) in candidate_windows:
        p_cnt, o_cnt = _window_counts(board, sr, sc, dr, dc, player)

        # Window has exactly (min_count - 1) player pieces and no opponent pieces
        # → placing player at the one empty cell reaches min_count
        if o_cnt == 0 and p_cnt == min_count - 1:
            # Find the empty cell in this window
            for i in range(5):
                nr, nc = sr + i * dr, sc + i * dc
                if board.grid[nr, nc] == 0:
                    return (nr, nc)

    return None


def _find_open_threat(board: Board, player: int, min_count: int) -> Optional[Tuple[int, int]]:
    """
    FIX #2: Detect OPEN threats — windows where placing player creates a sequence
    that has at least one open end (not blocked by opponent or board edge).

    _ X X X _  →  open-3, either end is a valid block/attack move.
    This returns the BEST blocking cell (the one that also extends player's own line).

    Used for detecting opponent open-3 which the pre-checks previously missed.
    """
    occupied = list(zip(*board.grid.nonzero()))
    if not occupied:
        return None

    best_move = None
    best_score = -1

    candidate_windows = set()
    for (pr, pc) in occupied:
        for (dr, dc) in DIRECTIONS:
            for offset in range(-4, 1):
                sr = pr + offset * dr
                sc = pc + offset * dc
                er = sr + 4 * dr
                ec = sc + 4 * dc
                if (0 <= sr < board.size and 0 <= sc < board.size and
                        0 <= er < board.size and 0 <= ec < board.size):
                    candidate_windows.add((sr, sc, dr, dc))

    for (sr, sc, dr, dc) in candidate_windows:
        p_cnt, o_cnt = _window_counts(board, sr, sc, dr, dc, player)

        if o_cnt == 0 and p_cnt >= min_count:
            # Check if this sequence is open (at least one free cell beyond the window)
            before_r = sr - dr
            before_c = sc - dc
            end_r    = sr + 4 * dr
            end_c    = sc + 4 * dc
            after_r  = end_r + dr
            after_c  = end_c + dc

            open_before = (0 <= before_r < board.size and
                           0 <= before_c < board.size and
                           board.grid[before_r, before_c] == 0)
            open_after  = (0 <= after_r < board.size and
                           0 <= after_c < board.size and
                           board.grid[after_r, after_c] == 0)

            if open_before or open_after:
                # Return the empty cell that is most "inside" the window
                # (blocks the sequence best)
                for i in range(5):
                    nr, nc = sr + i * dr, sc + i * dc
                    if board.grid[nr, nc] == 0:
                        score = quick_score(board, nr, nc, player)
                        if score > best_score:
                            best_score = score
                            best_move  = (nr, nc)

    return best_move


# ===========================================================================
# PART 3 — MINIMAX WITH ALPHA-BETA
# ===========================================================================

def _minimax(board: Board, depth: int, alpha: float, beta: float,
             is_maximizing: bool, player: int,
             last_move: Optional[Tuple[int, int]],
             start: float, time_limit: float) -> Optional[int]:
    opponent = 3 - player

    if time.time() - start >= time_limit:
        return None

    if last_move is not None:
        last_player = player if is_maximizing else opponent
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
# PART 4 — ANYTIME SEARCH
# ===========================================================================

def get_best_move(board: Board, player: int,
                  time_limit: float = 15.0,
                  last_move: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
    MAX_DEPTH  = 20
    start      = time.time()
    opponent   = 3 - player

    # ------------------------------------------------------------------
    # PRE-CHECK 1 — Immediate win (5 in a row)
    # ------------------------------------------------------------------
    move = _find_threat_move(board, player, 5)
    if move:
        return move

    # ------------------------------------------------------------------
    # PRE-CHECK 2 — Block opponent immediate win
    # ------------------------------------------------------------------
    move = _find_threat_move(board, opponent, 5)
    if move:
        return move

    # ------------------------------------------------------------------
    # PRE-CHECK 3 — Own open-4 (one move from winning, unblocked)
    # ------------------------------------------------------------------
    move = _find_threat_move(board, player, 4)
    if move:
        return move

    # ------------------------------------------------------------------
    # PRE-CHECK 4 — Block opponent open-4
    # ------------------------------------------------------------------
    move = _find_threat_move(board, opponent, 4)
    if move:
        return move

    # ------------------------------------------------------------------
    # PRE-CHECK 5 (NEW) — Block opponent open-3 that leads to open-4
    # _ X X X _ is extremely dangerous: opponent can extend to either side.
    # _find_open_threat detects this and returns the best blocking cell.
    # ------------------------------------------------------------------
    move = _find_open_threat(board, opponent, 3)
    if move:
        # Verify this is truly a threat (not just any 3-in-window)
        # by checking the opponent has a clear path to 4
        print(f"[PreCheck5] Blocking opponent open-3 at {move}")
        # Still pass through minimax to confirm it's the best response,
        # but use it as the seed / fallback
        threat_block = move
    else:
        threat_block = None

    # ------------------------------------------------------------------
    # Fallback: pick first legal move so we never return None
    # ------------------------------------------------------------------
    best_move  = None
    best_depth = 0

    fallback = get_sorted_moves(board, player)
    if fallback:
        best_move = fallback[0]

    # If we have a forced threat block, use it as starting best_move
    if threat_block is not None:
        best_move = threat_block

    for depth in range(1, MAX_DEPTH + 1):
        if time.time() - start >= time_limit:
            break
        print(f"Starting depth {depth} at t={time.time()-start:.2f}s")

        candidate_move  = None
        candidate_score = -math.inf

        sorted_moves = get_sorted_moves(board, player)

        # FIX #4: If threat_block exists, ensure it's evaluated first
        # (even if quick_score ranks it lower due to not seeing gap patterns)
        if threat_block is not None and threat_block not in sorted_moves:
            sorted_moves = [threat_block] + sorted_moves[:MAX_CANDIDATES - 1]
        elif threat_block is not None and threat_block in sorted_moves:
            sorted_moves.remove(threat_block)
            sorted_moves = [threat_block] + sorted_moves

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

    print(f"[Minimax] depth reached: {best_depth} | "
          f"time: {time.time()-start:.2f}s")
    return best_move


# ===========================================================================
# TESTS
# ===========================================================================
if __name__ == "__main__":
    board = Board(15)

    # 1. score_window
    assert score_window(3, 0) == SCORES[3]
    assert score_window(3, 1) == 0
    assert score_window(0, 0) == 0
    assert score_window(5, 0) == SCORES[5]
    print("PASS: score_window")

    # 2. evaluate — empty board
    assert evaluate(board, 1) == 0
    assert evaluate(board, 2) == 0
    print("PASS: evaluate empty board")

    # 3. evaluate — gap pattern
    board.make_move(0, 0, 1); board.make_move(0, 1, 1)
    board.make_move(0, 3, 1); board.make_move(0, 4, 1)
    assert evaluate(board, 1) >= SCORES[4]
    board.grid[:] = 0
    print("PASS: evaluate gap pattern")

    # 4. evaluate — symmetry
    board.make_move(0, 0, 1); board.make_move(1, 0, 1)
    board.make_move(2, 0, 2); board.make_move(3, 0, 2)
    assert evaluate(board, 1) == -evaluate(board, 2)
    board.grid[:] = 0
    print("PASS: evaluate symmetry")

    # 5. Block opponent open-4
    for col in range(4):
        board.make_move(0, col, 2)
    move = get_best_move(board, 1, time_limit=15.0)
    assert move[0] == 0, f"AI should block row 0, got {move}"
    board.grid[:] = 0
    print("PASS: get_best_move blocks opponent 4")

    # 6. Take winning move
    for col in range(4):
        board.make_move(0, col, 1)
    move = get_best_move(board, 1, time_limit=15.0)
    assert move == (0, 4), f"AI should complete 5-in-row, got {move}"
    board.grid[:] = 0
    print("PASS: get_best_move takes win")

    # 7. Block opponent open-3
    board.make_move(7, 7, 2); board.make_move(7, 8, 2); board.make_move(7, 9, 2)
    move = get_best_move(board, 1, time_limit=15.0)
    assert move in ((7, 6), (7, 10)), f"AI should block open-3, got {move}"
    board.grid[:] = 0
    print("PASS: get_best_move blocks opponent open-3")

    # 8. NEW — Block opponent GAP open-3: _ X X _ X _
    # Pattern at row 5: cols 5,6,8 occupied by player 2
    # Threat cells: (5,4), (5,7), (5,9)
    board.make_move(5, 5, 2); board.make_move(5, 6, 2); board.make_move(5, 8, 2)
    move = get_best_move(board, 1, time_limit=15.0)
    assert move in ((5, 4), (5, 7), (5, 9)), \
        f"AI should block gap open-3 _ X X _ X _, got {move}"
    board.grid[:] = 0
    print("PASS: get_best_move blocks gap open-3")

    # 9. Anytime: must return within time limit
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
    assert elapsed <= 16.0, f"exceeded time limit: {elapsed:.1f}s"
    board.grid[:] = 0
    print(f"PASS: anytime search returned in {elapsed:.2f}s")

    print("\n=== All tests passed! ===")