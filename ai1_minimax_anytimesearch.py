# -*- coding: utf-8 -*-
"""
ai1_minimax_anytimesearch.py — Gomoku AI (Medium/Hard mode)

Architecture:
  PRE-CHECK 1 : block_immediately   — O(1) local scan for opponent 4-in-a-row
  PRE-CHECK 2 : own 5-in-a-row win
  PRE-CHECK 3 : own open-4
  PRE-CHECK 4 : block opponent fork (critical)
  PRE-CHECK 5 : own fork
  MINIMAX     : iterative deepening with priority candidates injected at front
                (fork block / half-3 block / open-3 block are priority_candidates,
                 not hard bypasses — agent uses full time budget to choose best)

quick_score is mode-aware:
  must_defend=True  → keep defensive bonuses (opponent is dangerous)
  must_defend=False → offensive only; minimax handles defense naturally
"""

from game import Board
from typing import Tuple, Optional
import math
import time

# ---------------------------------------------------------------------------
SCORES = {1: 10, 2: 100, 3: 1_000, 4: 10_000, 5: 1_000_000}
WIN_SCORE  =  SCORES[5]
LOSE_SCORE = -SCORES[5]

DIRECTIONS = [(0,1),(1,0),(1,1),(1,-1)]


# ===========================================================================
# PART 1 — EVALUATION
# ===========================================================================

def score_window(p_cnt: int, o_cnt: int, open_ends: int = 0) -> int:
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
    opponent = 3 - player
    score    = 0
    for (dr, dc) in DIRECTIONS:
        for r in range(board.size):
            for c in range(board.size):
                er, ec = r + 4*dr, c + 4*dc
                if not (0 <= er < board.size and 0 <= ec < board.size):
                    continue
                p_cnt = o_cnt = 0
                for i in range(5):
                    cell = board.grid[r+i*dr, c+i*dc]
                    if cell == player:     p_cnt += 1
                    elif cell == opponent: o_cnt += 1
                br, bc = r-dr, c-dc
                ar, ac = r+5*dr, c+5*dc
                oe = 0
                if 0<=br<board.size and 0<=bc<board.size and board.grid[br,bc]==0: oe+=1
                if 0<=ar<board.size and 0<=ac<board.size and board.grid[ar,ac]==0: oe+=1
                score += score_window(p_cnt, o_cnt, oe)
                score -= score_window(o_cnt, p_cnt, oe)

    # Half-3 penalty — count opponent half-open 3s
    opp_half3_count = 0
    opp_half3_dirs  = 0
    seen_h = set()
    for (dr, dc) in DIRECTIONS:
        dir_has_half3 = False
        for r in range(board.size):
            for c in range(board.size):
                er, ec = r+4*dr, c+4*dc
                if not (0<=er<board.size and 0<=ec<board.size): continue
                key = (r, c, dr, dc)
                if key in seen_h: continue
                seen_h.add(key)
                o_cnt = p_cnt = 0
                for i in range(5):
                    cell = board.grid[r+i*dr, c+i*dc]
                    if cell == opponent: o_cnt += 1
                    elif cell == player: p_cnt += 1
                if o_cnt == 3 and p_cnt == 0:
                    br, bc = r-dr, c-dc
                    ar, ac = r+5*dr, c+5*dc
                    oe = 0
                    if 0<=br<board.size and 0<=bc<board.size and board.grid[br,bc]==0: oe+=1
                    if 0<=ar<board.size and 0<=ac<board.size and board.grid[ar,ac]==0: oe+=1
                    if oe == 1:
                        opp_half3_count += 1
                        dir_has_half3 = True
        if dir_has_half3:
            opp_half3_dirs += 1

    if opp_half3_dirs >= 3:
        score -= 25_000
    elif opp_half3_dirs >= 2:
        score -= 8_000
    score -= opp_half3_count * 15_000

    return score


# ===========================================================================
# PART 2 — MOVE ORDERING
# ===========================================================================

def _count_dir(board, r, c, dr, dc, player):
    count = 1
    for sign in (1,-1):
        nr, nc = r+sign*dr, c+sign*dc
        while 0<=nr<board.size and 0<=nc<board.size and board.grid[nr,nc]==player:
            count+=1; nr+=sign*dr; nc+=sign*dc
    return min(count, 5)


def quick_score(board: Board, r: int, c: int, player: int,
                must_defend: bool = True) -> int:
    DEFENSE_W = {5: 10.0, 4: 5.0, 3: 2.5, 2: 0.70, 1: 0.50}
    opponent  = 3 - player
    score     = 0

    for (dr, dc) in DIRECTIONS:
        board.grid[r,c] = player
        cnt_player = _count_dir(board, r, c, dr, dc, player)
        oe = 0
        for sign in (1,-1):
            nr, nc = r+sign*dr, c+sign*dc
            while 0<=nr<board.size and 0<=nc<board.size and board.grid[nr,nc]==player:
                nr+=sign*dr; nc+=sign*dc
            if 0<=nr<board.size and 0<=nc<board.size and board.grid[nr,nc]==0: oe+=1
        board.grid[r,c] = 0
        base = SCORES[cnt_player]
        score += base*2 if oe==2 else base if oe==1 else base//4

        board.grid[r,c] = opponent
        cnt_opp = _count_dir(board, r, c, dr, dc, opponent)
        board.grid[r,c] = 0
        score += DEFENSE_W[cnt_opp] * SCORES[cnt_opp]

    if must_defend:
        board.make_move(r, c, opponent)
        fours_opp, open3s_opp, half3s_opp, open2s_opp = _count_threats_after_move(board, r, c, opponent)
        board.undo_move(r, c)

        board.make_move(r, c, player)
        _, _, half3s_after, _ = _count_threats_after_move(board, r, c, opponent)
        board.undo_move(r, c)
        half3s_blocked = max(0, half3s_opp - half3s_after)

        if (fours_opp >= 2) or (fours_opp >= 1 and open3s_opp >= 1) or (open3s_opp >= 2):
            score += 50_000
        elif open3s_opp >= 1 and open2s_opp >= 2:
            score += 15_000
        elif open3s_opp >= 1 and open2s_opp >= 1:
            score += 5_000
        elif half3s_opp >= 3:
            score += 40_000
        elif half3s_opp >= 2:
            score += 15_000

        if half3s_blocked >= 2:
            score += 30_000
        elif half3s_blocked >= 1:
            score += 10_000

    return score


CANDIDATE_RADIUS = 2
MAX_CANDIDATES   = 8


def get_candidate_moves(board: Board) -> list:
    occupied = list(zip(*board.grid.nonzero()))
    if not occupied:
        center = board.size // 2
        return [(center, center)]
    candidates = set()
    for (r, c) in occupied:
        for dr in range(-CANDIDATE_RADIUS, CANDIDATE_RADIUS+1):
            for dc in range(-CANDIDATE_RADIUS, CANDIDATE_RADIUS+1):
                nr, nc = r+dr, c+dc
                if 0<=nr<board.size and 0<=nc<board.size and board.grid[nr,nc]==0:
                    candidates.add((nr,nc))
    return list(candidates)


def get_sorted_moves(board: Board, player: int,
                     depth_remaining: int = 99,
                     last_move: Optional[Tuple[int,int]] = None,
                     must_defend: bool = True,
                     priority_candidates: Optional[list] = None) -> list:
    if depth_remaining <= 1:   max_cand = 3
    elif depth_remaining <= 2: max_cand = 5
    else:                      max_cand = MAX_CANDIDATES

    # If priority candidates given, use them exclusively at top level
    if priority_candidates is not None:
        moves = list(priority_candidates)
    else:
        moves = get_candidate_moves(board)

    scored = []
    for m in moves:
        qs = quick_score(board, m[0], m[1], player, must_defend=must_defend)
        if last_move is not None:
            dist = abs(m[0]-last_move[0]) + abs(m[1]-last_move[1])
            qs  += max(0, (5-dist)) * 500
        scored.append((qs, m))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:max_cand]]


# ===========================================================================
# PART 2.4 — DANGER ASSESSMENT
# ===========================================================================

def _must_defend(board: Board, player: int,
                 last_move: Optional[Tuple[int,int]],
                 opponent: int) -> bool:
    """
    Returns True if opponent is in one of 4 dangerous states:
    1. block_immediately handles 5-in-a-row
    2. Opponent has open-3 → if we attack, they open-4 first
    3. Opponent 1 move away from fork (half-4 + open-3, or 2x half-4)
    4. Opponent has 3+ half-3 → can chain into fork in 2 moves
    """
    # State 2: opponent has open-3
    open3_cands = _get_open3_block_candidates(board, player)
    if open3_cands:
        return True

    # State 3: opponent 1 move from dangerous fork
    for (r, c) in get_candidate_moves(board):
        board.make_move(r, c, opponent)
        fours, open3s, half3s, open2s = _count_threats_after_move(board, r, c, opponent)
        board.undo_move(r, c)
        if fours >= 2 or (fours >= 1 and open3s >= 1):
            return True

    # State 4: opponent has 3+ half-3 from last move
    if last_move is not None:
        _, _, half3s_lm, _ = _count_threats_after_move(board, last_move[0], last_move[1], opponent)
        if half3s_lm >= 3:
            return True

    return False


# ===========================================================================
# PART 2.5 — THREAT DETECTION
# ===========================================================================

def _find_threat_move(board: Board, player: int, min_count: int) -> Optional[Tuple[int,int]]:
    opponent = 3 - player
    occupied = list(zip(*board.grid.nonzero()))
    if not occupied:
        return None
    seen = set()
    for (pr, pc) in occupied:
        for (dr, dc) in DIRECTIONS:
            for offset in range(-4, 1):
                sr, sc = pr+offset*dr, pc+offset*dc
                er, ec = sr+4*dr,      sc+4*dc
                if not (0<=sr<board.size and 0<=sc<board.size and
                        0<=er<board.size and 0<=ec<board.size):
                    continue
                key = (sr, sc, dr, dc)
                if key in seen: continue
                seen.add(key)
                p_cnt = o_cnt = 0
                for i in range(5):
                    cell = board.grid[sr+i*dr, sc+i*dc]
                    if cell == player:     p_cnt += 1
                    elif cell == opponent: o_cnt += 1
                if o_cnt == 0 and p_cnt == min_count - 1:
                    positions = [i for i in range(5) if board.grid[sr+i*dr, sc+i*dc] == player]
                    if positions[-1] - positions[0] != min_count - 2:
                        continue
                    for i in range(5):
                        nr, nc = sr+i*dr, sc+i*dc
                        if board.grid[nr,nc] == 0:
                            return (nr, nc)
    return None


def _get_open3_block_candidates(board: Board, player: int) -> list:
    opponent = 3 - player
    occupied = list(zip(*board.grid.nonzero()))
    if not occupied:
        return []
    final = set()
    seen = set()
    for (pr, pc) in occupied:
        for (dr, dc) in DIRECTIONS:
            for offset in range(-4, 1):
                sr, sc = pr+offset*dr, pc+offset*dc
                er, ec = sr+4*dr,      sc+4*dc
                if not (0<=sr<board.size and 0<=sc<board.size and
                        0<=er<board.size and 0<=ec<board.size):
                    continue
                key = (sr, sc, dr, dc)
                if key in seen: continue
                seen.add(key)
                o_cnt = p_cnt = 0
                for i in range(5):
                    cell = board.grid[sr+i*dr, sc+i*dc]
                    if cell == opponent:  o_cnt += 1
                    elif cell == player:  p_cnt += 1
                if o_cnt != 3 or p_cnt != 0:
                    continue
                seq  = [(sr+i*dr, sc+i*dc) for i in range(5)
                        if board.grid[sr+i*dr, sc+i*dc] == opponent]
                gaps = [(sr+i*dr, sc+i*dc) for i in range(5)
                        if board.grid[sr+i*dr, sc+i*dc] == 0]
                r0, c0 = seq[0]; r1, c1 = seq[-1]
                pr0, pc0 = int(r0-dr), int(c0-dc)
                pr1, pc1 = int(r1+dr), int(c1+dc)
                ob = (0<=pr0<board.size and 0<=pc0<board.size and board.grid[pr0,pc0]==0)
                oa = (0<=pr1<board.size and 0<=pc1<board.size and board.grid[pr1,pc1]==0)
                # Only truly open-3 (both ends free)
                if int(ob) + int(oa) < 2:
                    continue
                if ob: final.add((pr0, pc0))
                if oa: final.add((pr1, pc1))
                if len(gaps) == 1:
                    final.add((int(gaps[0][0]), int(gaps[0][1])))
    return list(final)


def _get_half3_block_candidates(board: Board, last_move: Tuple[int,int],
                                 opponent: int) -> list:
    player = 3 - opponent
    r0, c0 = last_move
    cands = set()
    for (dr, dc) in DIRECTIONS:
        for offset in range(5):
            sr, sc = r0-offset*dr, c0-offset*dc
            er, ec = sr+4*dr, sc+4*dc
            if not (0<=sr<board.size and 0<=sc<board.size and
                    0<=er<board.size and 0<=ec<board.size): continue
            p_cnt = o_cnt = 0
            for i in range(5):
                cell = board.grid[sr+i*dr, sc+i*dc]
                if cell == opponent: p_cnt += 1
                elif cell == player: o_cnt += 1
            if p_cnt != 3 or o_cnt != 0: continue
            seq  = [(sr+i*dr, sc+i*dc) for i in range(5)
                    if board.grid[sr+i*dr, sc+i*dc] == opponent]
            gaps = [(sr+i*dr, sc+i*dc) for i in range(5)
                    if board.grid[sr+i*dr, sc+i*dc] == 0]
            r1, c1 = seq[0]; r2, c2 = seq[-1]
            pr0, pc0 = int(r1-dr), int(c1-dc)
            pr1, pc1 = int(r2+dr), int(c2+dc)
            if 0<=pr0<board.size and 0<=pc0<board.size and board.grid[pr0,pc0]==0:
                cands.add((pr0, pc0))
            if 0<=pr1<board.size and 0<=pc1<board.size and board.grid[pr1,pc1]==0:
                cands.add((pr1, pc1))
            if len(gaps) == 1:
                cands.add((int(gaps[0][0]), int(gaps[0][1])))
    return list(cands)


# ===========================================================================
# PART 2.6 — FORK DETECTION
# ===========================================================================

def _count_threats_after_move(board: Board, r: int, c: int,
                               player: int) -> Tuple[int, int, int, int]:
    """Returns (four_threats, open3_threats, half_open3_threats, open2_threats).
    Local scan: only 5 windows per direction that contain (r,c). ~20x faster
    than full-board scan."""
    opponent = 3 - player
    four_threats = open3_threats = half_open3_threats = open2_threats = 0

    for (dr, dc) in DIRECTIONS:
        best_in_dir = 0  # 0=nothing 1=half3 2=open2 3=open3 4=four

        for offset in range(5):
            sr, sc = r - offset*dr, c - offset*dc
            er, ec = sr + 4*dr, sc + 4*dc
            if not (0<=sr<board.size and 0<=sc<board.size and
                    0<=er<board.size and 0<=ec<board.size):
                continue
            p_cnt = o_cnt = 0
            for i in range(5):
                cell = board.grid[sr+i*dr, sc+i*dc]
                if cell == player:     p_cnt += 1
                elif cell == opponent: o_cnt += 1
            if o_cnt > 0: continue

            if p_cnt == 4:
                best_in_dir = 4; break

            elif p_cnt == 3:
                br, bc = sr-dr, sc-dc
                ar, ac = sr+5*dr, sc+5*dc
                oe = 0
                if 0<=br<board.size and 0<=bc<board.size and board.grid[br,bc]==0: oe+=1
                if 0<=ar<board.size and 0<=ac<board.size and board.grid[ar,ac]==0: oe+=1
                if oe == 2:   best_in_dir = max(best_in_dir, 3)
                elif oe == 1: best_in_dir = max(best_in_dir, 1)

            elif p_cnt == 2:
                br, bc = sr-dr, sc-dc
                ar, ac = sr+5*dr, sc+5*dc
                oe = 0
                if 0<=br<board.size and 0<=bc<board.size and board.grid[br,bc]==0: oe+=1
                if 0<=ar<board.size and 0<=ac<board.size and board.grid[ar,ac]==0: oe+=1
                if oe == 2: best_in_dir = max(best_in_dir, 2)

        if best_in_dir == 4:   four_threats += 1
        elif best_in_dir == 3: open3_threats += 1
        elif best_in_dir == 2: open2_threats += 1
        elif best_in_dir == 1: half_open3_threats += 1

    return four_threats, open3_threats, half_open3_threats, open2_threats


def _is_fork_move(board: Board, r: int, c: int, player: int) -> bool:
    board.make_move(r, c, player)
    fours, open3s, half3s, open2s = _count_threats_after_move(board, r, c, player)
    board.undo_move(r, c)
    return (fours >= 2) or (fours >= 1 and open3s >= 1) or (open3s >= 2) or (open3s >= 1 and open2s >= 2)


def _is_critical_fork(board: Board, r: int, c: int, player: int) -> bool:
    board.make_move(r, c, player)
    fours, open3s, half3s, open2s = _count_threats_after_move(board, r, c, player)
    board.undo_move(r, c)
    return (fours >= 2) or (fours >= 1 and open3s >= 1) or (open3s >= 2)


def _find_fork_move(board: Board, player: int) -> Optional[Tuple[int,int]]:
    best_move  = None
    best_score = -1
    for (r, c) in get_candidate_moves(board):
        board.make_move(r, c, player)
        fours, open3s, half3s, open2s = _count_threats_after_move(board, r, c, player)
        board.undo_move(r, c)
        if (fours >= 2) or (fours >= 1 and open3s >= 1) or (open3s >= 2):
            fork_score = fours * 100 + open3s * 10
            if fork_score > best_score:
                best_score = fork_score
                best_move  = (r, c)
    return best_move


# ===========================================================================
# PART 3 — MINIMAX
# ===========================================================================

def _minimax(board, depth, alpha, beta, is_maximizing, player,
             last_move, start, time_limit, must_defend=True):
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

    cur_player = player if is_maximizing else opponent
    sorted_moves = get_sorted_moves(
        board, cur_player,
        depth_remaining=depth,
        must_defend=must_defend
    )

    if is_maximizing:
        best = -math.inf
        for (r, c) in sorted_moves:
            board.make_move(r, c, player)
            score = _minimax(board, depth-1, alpha, beta, False, player,
                             (r,c), start, time_limit, must_defend)
            board.undo_move(r, c)
            if score is None: return None
            best  = max(best, score)
            alpha = max(alpha, best)
            if beta <= alpha: break
        return best
    else:
        best = math.inf
        for (r, c) in sorted_moves:
            board.make_move(r, c, opponent)
            score = _minimax(board, depth-1, alpha, beta, True, player,
                             (r,c), start, time_limit, must_defend)
            board.undo_move(r, c)
            if score is None: return None
            best = min(best, score)
            beta = min(beta, best)
            if beta <= alpha: break
        return best


# ===========================================================================
# PART 3.5 — IMMEDIATE BLOCK
# ===========================================================================

def block_immediately(board: Board, last_move: Tuple[int,int],
                      player: int) -> Optional[Tuple[int,int]]:
    """O(1) local scan: block opponent 4-in-a-row (any pattern incl. scattered)."""
    opponent = 3 - player
    r0, c0 = last_move
    for (dr, dc) in DIRECTIONS:
        for offset in range(5):
            sr, sc = r0 - offset*dr, c0 - offset*dc
            er, ec = sr + 4*dr,      sc + 4*dc
            if not (0<=sr<board.size and 0<=sc<board.size and
                    0<=er<board.size and 0<=ec<board.size):
                continue
            o_cnt = p_cnt = 0
            empty = None
            for i in range(5):
                cell = board.grid[sr+i*dr, sc+i*dc]
                if cell == opponent:  o_cnt += 1
                elif cell == player:  p_cnt += 1
                else:                 empty = (sr+i*dr, sc+i*dc)
            if o_cnt == 4 and p_cnt == 0:
                return empty
    return None


# ===========================================================================
# PART 4 — ANYTIME SEARCH
# ===========================================================================

def get_best_move(board: Board, player: int,
                  time_limit: float = 20.0,
                  last_move: Optional[Tuple[int,int]] = None) -> Tuple[int,int]:
    MAX_DEPTH = 20
    start     = time.time()
    opponent  = 3 - player

    # PRE-CHECK 1: Block opponent immediate 4-in-a-row (O(1))
    if last_move is not None:
        move = block_immediately(board, last_move, player)
        if move: return move

    # PRE-CHECK 2: Own 5-in-a-row win
    move = _find_threat_move(board, player, 5)
    if move: return move

    # PRE-CHECK 3: Own open-4
    move = _find_threat_move(board, player, 4)
    if move: return move

    # PRE-CHECK 4: Own fork (critical — go immediately)
    move = _find_fork_move(board, player)
    if move:
        print(f"[Fork] own fork at {move}")
        return move

    # Assess danger level — determines quick_score mode and priority candidates
    defend = _must_defend(board, player, last_move, opponent)

    # Build priority candidates (injected at front of iterative deepening)
    priority_candidates = None

    if defend:
        # Block opponent fork first
        fork_block = _find_fork_move(board, opponent)
        if fork_block:
            priority_candidates = [fork_block]

        # If no critical fork, check half-3 >= 3
        if priority_candidates is None and last_move is not None:
            _, _, half3s_lm, _ = _count_threats_after_move(
                board, last_move[0], last_move[1], opponent)
            if half3s_lm >= 3:
                cands = _get_half3_block_candidates(board, last_move, opponent)
                if cands:
                    priority_candidates = cands

        # If still none, use open-3 block candidates
        if priority_candidates is None:
            open3_cands = _get_open3_block_candidates(board, player)
            if open3_cands:
                priority_candidates = open3_cands

    # MINIMAX — iterative deepening with full time budget
    best_move  = None
    best_depth = 0

    fallback = get_sorted_moves(board, player, must_defend=defend,
                                last_move=last_move,
                                priority_candidates=priority_candidates)
    if fallback:
        best_move = fallback[0]

    for depth in range(1, MAX_DEPTH+1):
        if time.time() - start >= time_limit:
            break

        candidate_move  = None
        candidate_score = -math.inf

        sorted_moves = get_sorted_moves(
            board, player,
            depth_remaining=depth,
            last_move=last_move,
            must_defend=defend,
            priority_candidates=priority_candidates if depth == 1 else None
        )
        timed_out = False

        for (r, c) in sorted_moves:
            if time.time() - start >= time_limit:
                timed_out = True; break

            board.make_move(r, c, player)
            score = _minimax(board, depth-1, -math.inf, math.inf,
                             False, player, (r,c), start, time_limit,
                             must_defend=defend)
            board.undo_move(r, c)

            if score is None:
                timed_out = True; break

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

    print(f"[Minimax] depth reached: {best_depth} | defend={defend} | time: {time.time()-start:.2f}s")
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
    print("PASS: score_window")

    # 2. evaluate empty
    assert evaluate(board, 1) == 0
    print("PASS: evaluate empty")

    # 3. open > blocked
    board.make_move(7,5,1); board.make_move(7,6,1); board.make_move(7,7,1)
    s_open = evaluate(board, 1)
    board.make_move(7,4,2)
    s_blocked = evaluate(board, 1)
    assert s_open > s_blocked
    board.grid[:] = 0
    print("PASS: open > blocked")

    # 4. Fork: double open-3
    board.make_move(7,5,1); board.make_move(7,6,1)
    board.make_move(5,7,1); board.make_move(6,7,1)
    assert _is_fork_move(board, 7, 7, 1), "Expected fork at (7,7)"
    board.grid[:] = 0
    print("PASS: _is_fork_move detects double open-3 fork")

    # 5. _find_fork_move finds it
    board.make_move(7,5,1); board.make_move(7,6,1)
    board.make_move(5,7,1); board.make_move(6,7,1)
    move = _find_fork_move(board, 1)
    assert move == (7,7), f"Expected (7,7), got {move}"
    board.grid[:] = 0
    print("PASS: _find_fork_move returns correct fork")

    # 6. Block opponent fork
    board.make_move(7,5,2); board.make_move(7,6,2)
    board.make_move(5,7,2); board.make_move(6,7,2)
    move = get_best_move(board, 1, time_limit=20.0)
    assert move == (7,7), f"Should block fork at (7,7), got {move}"
    board.grid[:] = 0
    print("PASS: blocks opponent fork")

    # 7. four+open3 fork
    board.make_move(7,4,1); board.make_move(7,5,1); board.make_move(7,6,1)
    board.make_move(5,8,1); board.make_move(6,8,1)
    assert _is_fork_move(board, 7, 8, 1), "Expected four+open3 fork at (7,8)"
    board.grid[:] = 0
    print("PASS: _is_fork_move detects four+open3 fork")

    # 8. Block win takes priority over fork
    board.make_move(0,0,2); board.make_move(0,1,2)
    board.make_move(0,2,2); board.make_move(0,3,2)
    board.make_move(7,5,1); board.make_move(7,6,1)
    board.make_move(5,7,1); board.make_move(6,7,1)
    move = get_best_move(board, 1, time_limit=20.0, last_move=(0,3))
    assert move in {(0,4), (0,5)}, f"Should block win first, got {move}"
    board.grid[:] = 0
    print("PASS: block win takes priority over own fork")

    # 9. Blocks opponent 4
    for col in range(4): board.make_move(0,col,2)
    move = get_best_move(board, 1, time_limit=20.0, last_move=(0,3))
    assert move[0] == 0
    board.grid[:] = 0
    print("PASS: blocks opponent 4")

    # 10. Takes win
    for col in range(4): board.make_move(0,col,1)
    move = get_best_move(board, 1, time_limit=20.0)
    assert move == (0,4)
    board.grid[:] = 0
    print("PASS: takes win")

    # 11. Blocks open-3
    board.make_move(7,7,2); board.make_move(7,8,2); board.make_move(7,9,2)
    move = get_best_move(board, 1, time_limit=20.0)
    assert move in ((7,6),(7,10))
    board.grid[:] = 0
    print("PASS: blocks open-3")

    # 12. _get_open3_block_candidates ignores half-open after block
    board.make_move(8,9,2); board.make_move(9,10,2); board.make_move(10,11,2)
    board.make_move(7,8,1)
    cands = _get_open3_block_candidates(board, 1)
    assert cands == [], f"Expected [], got {cands}"
    board.grid[:] = 0
    print("PASS: _get_open3_block_candidates ignores half-open after block")

    # 13. block_immediately catches scattered X X X _ X
    board.make_move(10,7,2); board.make_move(10,8,2)
    board.make_move(10,9,2); board.make_move(10,11,2)
    move = block_immediately(board, (10,11), 1)
    assert move == (10,10), f"Should block gap at (10,10), got {move}"
    board.grid[:] = 0
    print("PASS: block_immediately catches X X X _ X pattern")

    # 14. must_defend=False when no danger
    board.make_move(7,7,1); board.make_move(7,8,1)
    defend = _must_defend(board, 2, (7,8), 1)
    # Only 2 black pieces — no open-3, no fork threat, no half3>=3
    assert not defend, f"Expected no danger, got defend={defend}"
    board.grid[:] = 0
    print("PASS: _must_defend returns False when no danger")

    print("\n=== All tests passed! ===")