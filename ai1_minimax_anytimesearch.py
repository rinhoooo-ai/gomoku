# -*- coding: utf-8 -*-
"""
ai1_minimax_hard.py — Gomoku AI (Hard mode)

Changes vs medium (document 2):
  PRE-CHECK 6: own fork detection — nước tạo ra >=2 threats đồng thời → đi ngay
  PRE-CHECK 7: block opponent fork — nếu opponent có nước tạo fork → block ngay

Fork = sau khi đặt 1 quân, player có >=2 windows độc lập mà mỗi window
       chỉ cần thêm 1 quân nữa là thành 5 (tức là window có 4 quân, 0 đối thủ).
       Opponent không thể block cả 2 → guaranteed win trong 2 nước tiếp theo.

Cũng detect "soft fork": 1 four-threat + 1 open-three-threat (đủ nguy hiểm
để đi/block ngay thay vì để minimax tự tìm).
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
                    if cell == player:       p_cnt += 1
                    elif cell == opponent:   o_cnt += 1
                br, bc = r-dr, c-dc
                ar, ac = r+5*dr, c+5*dc
                oe = 0
                if 0<=br<board.size and 0<=bc<board.size and board.grid[br,bc]==0: oe+=1
                if 0<=ar<board.size and 0<=ac<board.size and board.grid[ar,ac]==0: oe+=1
                score += score_window(p_cnt, o_cnt, oe)
                score -= score_window(o_cnt, p_cnt, oe)
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


def quick_score(board: Board, r: int, c: int, player: int) -> int:
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
    
    board.make_move(r, c, opponent)
    fours, open3s, half3s, open2s = _count_threats_after_move(board, r, c, opponent)
    board.undo_move(r, c)
    
    if (fours >= 2) or (fours >= 1 and open3s >= 1) or (open3s >= 2):
        score += 50_000   # critical fork, ưu tiên gần như tuyệt đối
    elif open3s >= 1 and open2s >= 2:
        score += 15_000   # soft fork, nguy hiểm hơn normal move nhiều
    elif open3s >= 1 and open2s >= 1:
        score += 5_000

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
                     last_move: Optional[Tuple[int,int]] = None) -> list:
    if depth_remaining <= 1:   max_cand = 3
    elif depth_remaining <= 2: max_cand = 5
    else:                      max_cand = MAX_CANDIDATES

    moves = get_candidate_moves(board)

    def sort_key(move):
        qs = quick_score(board, move[0], move[1], player)
        if last_move is not None:
            dist = abs(move[0]-last_move[0]) + abs(move[1]-last_move[1])
            qs  += max(0, (5-dist)) * 500
        return qs

    moves = sorted(moves, key=sort_key, reverse=True)
    return moves[:max_cand]


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
                    if cell == player:    p_cnt += 1
                    elif cell == opponent: o_cnt += 1
                if o_cnt == 0 and p_cnt == min_count - 1:
                    positions = [i for i in range(5) if board.grid[sr+i*dr, sc+i*dc] == player]
                    if positions[-1] - positions[0] != min_count - 2:
                        continue  # scattered, skip
                    for i in range(5):
                        nr, nc = sr+i*dr, sc+i*dc
                        if board.grid[nr,nc] == 0:
                            return (nr, nc)
    return None


def _find_open3_block(board: Board, player: int) -> Optional[Tuple[int,int]]:
    opponent = 3 - player
    occupied = list(zip(*board.grid.nonzero()))
    if not occupied:
        return None
    seen = set()
    both_open_moves = []
    one_open_moves  = []
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
                br, bc = sr-dr, sc-dc
                ar, ac = sr+5*dr, sc+5*dc
                ob = (0<=br<board.size and 0<=bc<board.size and board.grid[br,bc]==0)
                oa = (0<=ar<board.size and 0<=ac<board.size and board.grid[ar,ac]==0)
                oe = int(ob) + int(oa)
                if oe == 0:
                    continue

                seq = [(sr+i*dr, sc+i*dc) for i in range(5) if board.grid[sr+i*dr, sc+i*dc] == opponent]
                gaps = [(sr+i*dr, sc+i*dc) for i in range(5) if board.grid[sr+i*dr, sc+i*dc] == 0]

                r0, c0 = seq[0]
                r1, c1 = seq[-1]

                candidates = []
                pr0, pc0 = int(r0-dr), int(c0-dc)
                pr1, pc1 = int(r1+dr), int(c1+dc)
                if 0<=pr0<board.size and 0<=pc0<board.size and board.grid[pr0,pc0]==0:
                    candidates.append((pr0, pc0))
                if 0<=pr1<board.size and 0<=pc1<board.size and board.grid[pr1,pc1]==0:
                    candidates.append((pr1, pc1))

                if len(gaps) == 1:
                    candidates.extend(gaps)

                if not candidates:
                    continue
                best = max(candidates, key=lambda m: quick_score(board, m[0], m[1], player))
                qs   = quick_score(board, best[0], best[1], player)
                if oe == 2: both_open_moves.append((qs, best))
                else:        one_open_moves.append((qs, best))

    if both_open_moves:
        return max(both_open_moves, key=lambda x: x[0])[1]
    if one_open_moves:
        return max(one_open_moves,  key=lambda x: x[0])[1]
    return None


# ===========================================================================
# PART 2.6 — FORK DETECTION  (NEW)
# ===========================================================================

def _count_threats_after_move(board: Board, r: int, c: int, player: int) -> Tuple[int, int, int, int]:
    """Returns (four_threats, open3_threats, half_open3_threats, open2_threats)"""
    opponent = 3 - player
    four_threats = 0
    open3_threats = 0
    half_open3_threats = 0
    open2_threats = 0

    for (dr, dc) in DIRECTIONS:
        best_in_dir = 0  # 0=nothing, 1=half_open3, 2=open2, 3=open3, 4=four

        for offset in range(-4, 1):
            sr, sc = r + offset*dr, c + offset*dc
            er, ec = sr + 4*dr, sc + 4*dc
            if not (0<=sr<board.size and 0<=sc<board.size and
                    0<=er<board.size and 0<=ec<board.size):
                continue

            p_cnt = o_cnt = 0
            for i in range(5):
                cell = board.grid[sr+i*dr, sc+i*dc]
                if cell == player:     p_cnt += 1
                elif cell == opponent: o_cnt += 1

            if o_cnt > 0:
                continue

            if p_cnt == 4:
                best_in_dir = 4
                break

            elif p_cnt == 3:
                br, bc = sr-dr, sc-dc
                ar, ac = sr+5*dr, sc+5*dc
                oe = 0
                if 0<=br<board.size and 0<=bc<board.size and board.grid[br,bc]==0: oe+=1
                if 0<=ar<board.size and 0<=ac<board.size and board.grid[ar,ac]==0: oe+=1
                if oe == 2:
                    best_in_dir = max(best_in_dir, 3)   # open3
                elif oe == 1:
                    best_in_dir = max(best_in_dir, 1)   # half_open3

            elif p_cnt == 2:
                br, bc = sr-dr, sc-dc
                ar, ac = sr+5*dr, sc+5*dc
                oe = 0
                if 0<=br<board.size and 0<=bc<board.size and board.grid[br,bc]==0: oe+=1
                if 0<=ar<board.size and 0<=ac<board.size and board.grid[ar,ac]==0: oe+=1
                if oe == 2:
                    best_in_dir = max(best_in_dir, 2)   # open2
                elif oe == 1:
                    pass  # half_open2 — bỏ qua, không đủ nguy hiểm

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
             last_move, start, time_limit):
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
            score = _minimax(board, depth-1, alpha, beta, False, player, (r,c), start, time_limit)
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
            score = _minimax(board, depth-1, alpha, beta, True, player, (r,c), start, time_limit)
            board.undo_move(r, c)
            if score is None: return None
            best = min(best, score)
            beta = min(beta, best)
            if beta <= alpha: break
        return best


# ===========================================================================
# PART 4 — ANYTIME SEARCH
# ===========================================================================

def get_best_move(board: Board, player: int,
                  time_limit: float = 20.0,
                  last_move: Optional[Tuple[int,int]] = None) -> Tuple[int,int]:
    MAX_DEPTH = 20
    start     = time.time()
    opponent  = 3 - player

    # PRE-CHECK 1: Immediate win
    move = _find_threat_move(board, player, 5)
    if move: return move

    # PRE-CHECK 2: Block opponent immediate win
    move = _find_threat_move(board, opponent, 5)
    if move: return move

    # PRE-CHECK 3: Own open-4
    move = _find_threat_move(board, player, 4)
    if move: return move

    # PRE-CHECK 5: Block opponent fork
    move = _find_fork_move(board, opponent)
    if move:
        print(f"[Fork] blocking opponent fork at {move}")
        return move

    # PRE-CHECK 6: Block opponent open-3
    print(f"Player {player} thinking...")
    move = _find_open3_block(board, player)
    print(f"  open3 block found: {move}")
    if move: return move

    # PRE-CHECK 7: Own fork
    move = _find_fork_move(board, player)
    if move:
        print(f"[Fork] own fork at {move}")
        return move

    # MINIMAX
    best_move  = None
    best_depth = 0

    fallback = get_sorted_moves(board, player, last_move=last_move)
    if fallback:
        best_move = fallback[0]

    for depth in range(1, MAX_DEPTH+1):
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
                timed_out = True; break

            board.make_move(r, c, player)
            score = _minimax(board, depth-1, -math.inf, math.inf,
                             False, player, (r,c), start, time_limit)
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

    # 4. _count_threats_after_move: double open-3
    board.make_move(7,6,1); board.make_move(7,7,1)   # horiz open-2
    board.make_move(6,7,1); board.make_move(5,7,1)   # vert open-2
    board.make_move(7,8,1)                            # extend horiz to open-3
    board.make_move(r:=4,c:=7,player:=1)              # extend vert to open-3
    board.undo_move(4,7)
    # place at (7,5): should create open-3 horiz + open-3 vert? no—just test fork detection
    board.grid[:] = 0

    # 5. Fork: _ X X _ + _ X X _ crossing → (7,7) creates double open-3
    board.make_move(7,5,1); board.make_move(7,6,1)   # horiz open-2
    board.make_move(5,7,1); board.make_move(6,7,1)   # vert open-2
    assert _is_fork_move(board, 7, 7, 1), "Expected fork at (7,7)"
    board.grid[:] = 0
    print("PASS: _is_fork_move detects double open-3 fork")

    # 6. _find_fork_move finds it
    board.make_move(7,5,1); board.make_move(7,6,1)
    board.make_move(5,7,1); board.make_move(6,7,1)
    move = _find_fork_move(board, 1)
    assert move == (7,7), f"Expected (7,7), got {move}"
    board.grid[:] = 0
    print("PASS: _find_fork_move returns correct fork")

    # 7. Block opponent fork
    board.make_move(7,5,2); board.make_move(7,6,2)
    board.make_move(5,7,2); board.make_move(6,7,2)
    move = get_best_move(board, 1, time_limit=20.0)
    assert move == (7,7), f"Should block fork at (7,7), got {move}"
    board.grid[:] = 0
    print("PASS: blocks opponent fork")

    # 8. four+open3 fork: X X X _ horiz + X X open-2 vert
    board.make_move(7,4,1); board.make_move(7,5,1); board.make_move(7,6,1)  # 3-in-row horiz
    board.make_move(5,8,1); board.make_move(6,8,1)                          # open-2 vert
    # (7,8) creates: horiz open-4 (7,4..7,8) + vert open-3 via (5,8)(6,8)(7,8)
    assert _is_fork_move(board, 7, 8, 1), "Expected four+open3 fork at (7,8)"
    board.grid[:] = 0
    print("PASS: _is_fork_move detects four+open3 fork")

    # 9. Block win still takes priority over fork
    board.make_move(0,0,2); board.make_move(0,1,2)
    board.make_move(0,2,2); board.make_move(0,3,2)  # opponent 4-in-row
    board.make_move(7,5,1); board.make_move(7,6,1)
    board.make_move(5,7,1); board.make_move(6,7,1)  # own fork available
    move = get_best_move(board, 1, time_limit=20.0)
    assert move in {(0,4), (0,5)}, f"Should block win first, got {move}"
    board.grid[:] = 0
    print("PASS: block win takes priority over own fork")

    # 10. Standard tests
    for col in range(4): board.make_move(0,col,2)
    move = get_best_move(board, 1, time_limit=20.0)
    assert move[0] == 0
    board.grid[:] = 0
    print("PASS: blocks opponent 4")

    for col in range(4): board.make_move(0,col,1)
    move = get_best_move(board, 1, time_limit=20.0)
    assert move == (0,4)
    board.grid[:] = 0
    print("PASS: takes win")

    board.make_move(7,7,2); board.make_move(7,8,2); board.make_move(7,9,2)
    move = get_best_move(board, 1, time_limit=20.0)
    assert move in ((7,6),(7,10))
    board.grid[:] = 0
    print("PASS: blocks open-3")

    print("\n=== All tests passed! ===")