from game import Board
from typing import Tuple, Optional
import math
import time
import random

# ---------------------------------------------------------------------------
# HYBRID CONSTANTS
# ---------------------------------------------------------------------------
C_PUCT           = 1.414   # UCB exploration constant
TOP_K            = 5       # top MCTS candidates passed to minimax phase
CANDIDATE_RADIUS = 2
MAX_CANDIDATES   = 10
MAX_ROLLOUT      = 30

# Time split: 40% MCTS exploration, 60% minimax refinement.
# Minimax gets more time because its signal is more precise.
PHASE1_RATIO     = 0.4
PHASE2_RATIO     = 0.6

# Scoring constants (same as minimax agent)
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
# PART 1 — EVALUATION  (shared with minimax agent)
# ===========================================================================

def score_window(p_cnt: int, o_cnt: int) -> int:
    """Score a single 5-cell window for ONE player."""
    if p_cnt > 0 and o_cnt == 0:
        return SCORES[p_cnt]
    return 0


def evaluate(board: Board, player: int) -> int:
    """
    Heuristic board evaluation for `player`.
    Scans every 5-cell window in all 4 directions.
    Returns positive values for player advantage, negative for opponent.
    """
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
# PART 2 — MOVE ORDERING  (shared with minimax agent)
# ===========================================================================

def _count_dir(board: Board, r: int, c: int,
               dr: int, dc: int, player: int) -> int:
    """
    Count consecutive `player` pieces through (r, c) in direction (dr, dc),
    counting both ways (sign = +1 and sign = -1).
    """
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
    Fast heuristic score for placing `player` at empty cell (r, c).
    Used for move ordering — higher = more promising.

    Tentatively places each player to count consecutive runs, then
    applies defense weights so blocking threats scores above pure attack.

    Defense weights:
        5 → 2.0  (immediate win threat)
        4 → 1.5  (one move from win)
        3 → 1.2  (open-3 block scores above building own open-3)
        2 → 0.70
        1 → 0.50
    """
    DEFENSE_W = {5: 2.0, 4: 1.5, 3: 1.2, 2: 0.70, 1: 0.50}
    opponent  = 3 - player
    score     = 0

    for (dr, dc) in DIRECTIONS:
        board.grid[r, c] = player
        cnt_player = _count_dir(board, r, c, dr, dc, player)
        board.grid[r, c] = 0

        board.grid[r, c] = opponent
        cnt_opponent = _count_dir(board, r, c, dr, dc, opponent)
        board.grid[r, c] = 0

        score += SCORES[cnt_player]
        score += DEFENSE_W[cnt_opponent] * SCORES[cnt_opponent]

    return score


def get_candidate_moves(board: Board) -> list:
    """
    Return empty cells within CANDIDATE_RADIUS of any occupied cell.
    If the board is empty, return the center cell only.
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


def get_sorted_moves(board: Board, player: int,
                     last_move: Optional[Tuple[int, int]] = None) -> list:
    """
    Return candidate cells sorted by quick_score (descending), capped at
    MAX_CANDIDATES. If last_move is provided, nearby cells get a proximity
    bonus so direct responses are examined first.
    """
    moves = get_candidate_moves(board)

    def sort_key(move):
        qs = quick_score(board, move[0], move[1], player)
        if last_move is not None:
            dist = abs(move[0] - last_move[0]) + abs(move[1] - last_move[1])
            qs  += max(0, (5 - dist)) * 500
        return qs

    moves = sorted(moves, key=sort_key, reverse=True)
    return moves[:MAX_CANDIDATES]


# ===========================================================================
# PART 2.5 — EXPLICIT THREAT CHECKS  (shared with minimax agent)
# ===========================================================================

def _find_threat_move(board: Board, player: int, min_count: int,
                      require_open: bool = False) -> Optional[Tuple[int, int]]:
    """
    Return the first candidate cell where placing `player` creates a run of
    >= min_count. If require_open=True, skip runs where both ends are blocked
    (edge or opponent stone) — edge-blocked runs are weak threats.
    """
    for (r, c) in get_candidate_moves(board):
        board.grid[r, c] = player
        for (dr, dc) in DIRECTIONS:
            cnt = _count_dir(board, r, c, dr, dc, player)
            if cnt >= min_count:
                if require_open:
                    open_ends = 0
                    for sign in (1, -1):
                        nr, nc = r + sign * dr, c + sign * dc
                        while (0 <= nr < board.size and
                               0 <= nc < board.size and
                               board.grid[nr, nc] == player):
                            nr += sign * dr
                            nc += sign * dc
                        if (0 <= nr < board.size and
                                0 <= nc < board.size and
                                board.grid[nr, nc] == 0):
                            open_ends += 1
                    if open_ends == 0:
                        continue
                board.grid[r, c] = 0
                return (r, c)
        board.grid[r, c] = 0
    return None


def _pre_checks(board: Board, player: int) -> Optional[Tuple[int, int]]:
    """
    Run all forced-response checks before search.
    Returns a move immediately if any forced condition is met, else None.

    Priority order:
        1. Immediate win
        2. Immediate block (opponent wins next move)
        3. Own open-4 — push to 5 next turn
        4. Opponent fork — block cell that creates 2+ threats simultaneously
        5. Opponent open-3 with no counter-threat of our own
    """
    opponent = 3 - player

    # 1. Immediate win
    move = _find_threat_move(board, player, 5)
    if move:
        return move

    # 2. Immediate block
    move = _find_threat_move(board, opponent, 5)
    if move:
        return move

    # 3. Own open-4
    move = _find_threat_move(board, player, 4)
    if move:
        return move

    # 4. Opponent fork — one cell creates 2+ directions of count >= 4
    for (r, c) in get_candidate_moves(board):
        board.grid[r, c] = opponent
        threat_count = sum(
            1 for (dr, dc) in DIRECTIONS
            if _count_dir(board, r, c, dr, dc, opponent) >= 4
        )
        board.grid[r, c] = 0
        if threat_count >= 2:
            return (r, c)

    # 5. Opponent open-3, only block if we have no counter open-3
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

    my_open3 = _find_threat_move(board, player, 3, require_open=True)
    if threat_cells and not my_open3:
        return max(threat_cells,
                   key=lambda m: quick_score(board, m[0], m[1], player))

    return None


# ===========================================================================
# PART 3 — MCTS NODE
# ===========================================================================

class MCTSNode:
    """
    MCTS tree node for Phase 1 exploration.

    untried_moves is pre-sorted by quick_score so MCTS expands promising
    moves first, producing higher-quality top-K candidates for Phase 2.
    """

    def __init__(self, board: Board, player: int,
                 parent: Optional["MCTSNode"] = None,
                 move: Optional[Tuple[int, int]] = None):
        self.board    = board
        self.player   = player
        self.parent   = parent
        self.move     = move
        self.children : list["MCTSNode"] = []
        self.visits   : int   = 0
        self.wins     : float = 0.0

        # Pre-sort untried moves by quick_score so expand() tries best first.
        candidates = get_candidate_moves(board)[:MAX_CANDIDATES]
        self.untried_moves = sorted(
            candidates,
            key=lambda m: quick_score(board, m[0], m[1], player),
            reverse=True
        )

    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    def is_terminal(self) -> bool:
        if self.move is None:
            return False
        last_player = 3 - self.player   # player who just moved
        return self.board.check_win(self.move[0], self.move[1], last_player) or self.board.is_full()

    def ucb_score(self, c: float = C_PUCT) -> float:
        if self.visits == 0:
            return math.inf
        return (self.wins / self.visits) + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def best_child(self, c: float = C_PUCT) -> "MCTSNode":
        return max(self.children, key=lambda child: child.ucb_score(c))

    def expand(self) -> "MCTSNode":
        """Pop the highest-scored untried move and create a child node."""
        move = self.untried_moves.pop(0)   # pop front — best move first
        move = (int(move[0]), int(move[1]))
        new_board = self.board.copy()
        new_board.make_move(move[0], move[1], self.player)
        child = MCTSNode(new_board, 3 - self.player, self, move)
        self.children.append(child)
        return child

    def backpropagate(self, result: float) -> None:
        self.visits += 1
        self.wins   += result
        if self.parent:
            self.parent.backpropagate(1 - result)


# ===========================================================================
# PART 4 — EVALUATE-BASED ROLLOUT
# Replaces random simulation with a shallow heuristic evaluation.
# Much faster signal than rolling out to end-of-game.
# ===========================================================================

def eval_rollout(board: Board, player: int) -> float:
    """
    Replace random rollout with a normalized heuristic evaluation.

    Instead of simulating to game end (slow, high variance), evaluate the
    current board position directly with evaluate() and normalize to [0, 1]
    using tanh so the result is compatible with MCTS backpropagation.

        raw  = evaluate(board, player)
        result = 0.5 + 0.5 * tanh(raw / WIN_SCORE)

    This gives:
        strong advantage  →  result close to 1.0
        even position     →  result close to 0.5
        losing position   →  result close to 0.0

    Much cheaper per call than rollout() and carries more signal because
    evaluate() scans the full board rather than one random playout path.

    INPUT:  board, player
    OUTPUT: float ∈ (0, 1)
    """
    raw = evaluate(board, player)
    return 0.5 + 0.5 * math.tanh(raw / 10000)  # scale by SCORES[4] for better signal separation


# ===========================================================================
# PART 5 — PHASE 1: MCTS EXPLORATION
# ===========================================================================

def mcts_phase(board: Board, player: int, time_budget: float) -> list:
    """
    Run MCTS for time_budget seconds using eval_rollout (not random rollout).
    Returns top-K children of root sorted by visit count (descending).

    Key improvements over vanilla MCTS:
        - untried_moves pre-sorted by quick_score → expand best first
        - eval_rollout replaces random simulation → cheaper, more accurate
    """
    root  = MCTSNode(board.copy(), player)
    start = time.time()
    iterations = 0

    while time.time() - start < time_budget:
        # 1. Selection
        node = root
        while node.is_fully_expanded() and not node.is_terminal():
            node = node.best_child()

        # 2. Expansion
        if not node.is_terminal():
            node = node.expand()
            # 3. Simulation — eval-based, not random rollout
            result = eval_rollout(node.board, node.player)
            # 4. Backpropagation
            node.backpropagate(result)
        else:
            node.backpropagate(0.5)

        iterations += 1

    print(f"[Phase 1] MCTS iterations: {iterations}")
    # Select by win rate (quality) not visit count — eval_rollout signal
    # is non-random so high visits don't guarantee high quality.
    return sorted(root.children,
                  key=lambda c: c.wins / c.visits if c.visits > 0 else 0,
                  reverse=True)[:TOP_K]


# ===========================================================================
# PART 6 — MINIMAX WITH ALPHA-BETA  (shared with minimax agent)
# ===========================================================================

def _minimax(board: Board, depth: int, alpha: float, beta: float,
             is_maximizing: bool, player: int,
             last_move: Optional[Tuple[int, int]],
             start: float, time_limit: float) -> Optional[int]:
    """
    Minimax with Alpha-Beta Pruning + time guard.
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

    sorted_moves = get_sorted_moves(board, player if is_maximizing else opponent)

    if is_maximizing:
        best = -math.inf
        for (r, c) in sorted_moves:
            board.make_move(r, c, player)
            score = _minimax(board, depth - 1, alpha, beta, False, player, (r, c), start, time_limit)
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
# PART 7 — PHASE 2: MINIMAX REFINEMENT
# ===========================================================================

def minimax_phase(board: Board, player: int,
                  top_k_nodes: list,
                  time_budget: float) -> Tuple[int, int]:
    """
    Run iterative-deepening minimax on each top-K MCTS candidate.
    Returns the move with the highest refined minimax score.

    time_budget is split equally among top_k_nodes.
    """
    MAX_DEPTH = 20

    if not top_k_nodes:
        return get_candidate_moves(board)[0]

    time_per_node    = time_budget / len(top_k_nodes)
    best_overall     = -math.inf
    best_move        = top_k_nodes[0].move

    for node in top_k_nodes:
        node_best  = -math.inf
        start_node = time.time()

        for depth in range(1, MAX_DEPTH + 1):
            if time.time() - start_node >= time_per_node:
                break
            score = _minimax(node.board, depth, -math.inf, math.inf,
                             False, player, node.move, start_node, time_per_node)
            if score is None:
                break
            node_best = score

        print(f"[Phase 2] move {node.move} → minimax score {node_best}")

        if node_best > best_overall:
            best_overall = node_best
            best_move    = node.move

    return best_move


# ===========================================================================
# PART 8 — HYBRID SEARCH  (orchestrates Phase 1 + Phase 2)
# ===========================================================================

def hybrid_search(board: Board, player: int,
                  time_limit: float = 15.0,
                  last_move: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
    """
    Hybrid MCTS + Minimax search.

    Pipeline:
        1. Pre-checks  — handle forced responses instantly (win/block/fork).
        2. Phase 1     — MCTS exploration     (PHASE1_RATIO of time_limit).
        3. Phase 2     — Minimax refinement   (PHASE2_RATIO of time_limit).

    MCTS explores broadly with eval_rollout signal; Minimax sharpens the
    top-K candidates with deep iterative-deepening search.
    """
    # 1. Forced responses — same pre-checks as medium agent
    move = _pre_checks(board, player)
    if move:
        return move

    phase1_budget = time_limit * PHASE1_RATIO
    phase2_budget = time_limit * PHASE2_RATIO

    # 2. MCTS exploration
    top_k = mcts_phase(board, player, phase1_budget)
    if not top_k:
        fallback = get_sorted_moves(board, player, last_move)
        return fallback[0] if fallback else get_candidate_moves(board)[0]

    # 3. Minimax refinement
    return minimax_phase(board, player, top_k, phase2_budget)


# ===========================================================================
# TESTS
# ===========================================================================
if __name__ == "__main__":
    board = Board(15)

    # evaluate
    assert evaluate(board, 1) == 0
    assert evaluate(board, 2) == 0
    board.make_move(0, 0, 1); board.make_move(1, 0, 1)
    board.make_move(2, 0, 2); board.make_move(3, 0, 2)
    assert evaluate(board, 1) == -evaluate(board, 2)
    board.grid[:] = 0
    print("PASS: evaluate")

    # eval_rollout range
    board.make_move(7, 7, 1)
    result = eval_rollout(board.copy(), player=1)
    assert 0.0 <= result <= 1.0
    board.grid[:] = 0
    print("PASS: eval_rollout range")

    # eval_rollout — winning position > 0.5
    for col in range(4):
        board.make_move(0, col, 1)
    result = eval_rollout(board.copy(), player=1)
    assert result > 0.5, f"4-in-row should score > 0.5, got {result}"
    board.grid[:] = 0
    print("PASS: eval_rollout winning position")

    # pre_checks — immediate win
    for col in range(4):
        board.make_move(0, col, 1)
    move = _pre_checks(board, 1)
    assert move == (0, 4), f"should find immediate win, got {move}"
    board.grid[:] = 0
    print("PASS: _pre_checks immediate win")

    # pre_checks — immediate block
    for col in range(4):
        board.make_move(0, col, 2)
    move = _pre_checks(board, 1)
    assert move == (0, 4), f"should block opponent win, got {move}"
    board.grid[:] = 0
    print("PASS: _pre_checks immediate block")

    # hybrid_search — take winning move
    for col in range(4):
        board.make_move(0, col, 1)
    move = hybrid_search(board, player=1, time_limit=15.0)
    assert move == (0, 4), f"should complete 5-in-row, got {move}"
    board.grid[:] = 0
    print("PASS: hybrid_search takes win")

    # hybrid_search — block opponent open-4
    for col in range(4):
        board.make_move(0, col, 2)
    move = hybrid_search(board, player=1, time_limit=15.0)
    assert move == (0, 4), f"should block row 0, got {move}"
    board.grid[:] = 0
    print("PASS: hybrid_search blocks opponent 4")

    # hybrid_search — returns within time limit
    random.seed(42)
    for _ in range(10):
        r, c = random.randint(5, 9), random.randint(5, 9)
        if board.grid[r, c] == 0:
            board.make_move(r, c, random.choice([1, 2]))
    t0      = time.time()
    move    = hybrid_search(board, player=1, time_limit=15.0)
    elapsed = time.time() - t0
    assert move is not None
    assert elapsed <= 16.0, f"exceeded time: {elapsed:.1f}s"
    board.grid[:] = 0
    print(f"PASS: hybrid_search returned in {elapsed:.2f}s")

    print("\n=== All tests passed! ===")