# Gomoku AI — Three Search Paradigms

**[Play Live →](https://rinhoooo-ai.github.io/gomoku)**

A fully client-side Gomoku engine implementing three distinct AI search paradigms from scratch, running in-browser via Pyodide (no server required).

---

## Why This Project

Most game AI demos wrap a library or tune hyperparameters. This project builds three fundamentally different search algorithms from scratch and pits them against each other at increasing difficulty levels — making the algorithmic tradeoffs tangible and playable.

---

## AI Agents

### Easy — Pure MCTS (`ai2_mcts.py`)
Monte Carlo Tree Search with heuristic rollouts. Uses UCB1 for tree policy and a win/block-aware simulation policy instead of pure random play. ~174 iterations per move at 15s budget.

**Limitation by design:** without a value network, rollout signal is noisy. MCTS struggles with tactical sharp positions — which is exactly why it's the easy level.

### Medium — Minimax + Alpha-Beta + Iterative Deepening (`ai1_minimax_anytimesearch.py`)
Anytime search via iterative deepening: completes depth *d* fully before starting depth *d+1*, so the best result from the last completed depth is always available when time runs out. Alpha-beta pruning with move ordering (heuristic `quick_score`) achieves effective branching factor ~√N, reaching depth 5–6 on a 15×15 board within 15s.

Key optimizations:
- Candidate pruning: only considers cells within radius-2 of occupied stones (~15 candidates vs 225)
- Move ordering: sorts candidates by `quick_score` before search so alpha-beta prunes aggressively
- Early termination: exits immediately on `WIN_SCORE` without exhausting the time budget

### Hard — Hybrid MCTS + Minimax (`ai3_minimax_anytimesearch_mcts.py`)
Two-phase pipeline exploiting the complementary strengths of both paradigms:

**Phase 1 (7.5s) — MCTS Exploration:** Builds a search tree using heuristic rollouts. Returns the top-3 candidate moves by visit count — MCTS is good at broad exploration and identifying promising regions of the search space.

**Phase 2 (7.5s) — Minimax Refinement:** Runs iterative-deepening minimax on each of the 3 MCTS candidates (~2.5s per candidate), reaching depth 5+. Minimax is good at precise tactical evaluation — it sharpens the coarse MCTS signal into a reliable score.

Final move = candidate with highest minimax score after refinement.

> *"MCTS explores, Minimax exploits."*

---

## Technical Stack

| Layer | Technology |
|---|---|
| AI Engine | Python (NumPy) |
| Browser Runtime | Pyodide — CPython compiled to WebAssembly |
| Frontend | Vanilla HTML/CSS/JS, Canvas API |
| Hosting | GitHub Pages — serverless, 24/7, zero cost |

No backend. No API calls. The entire AI runs inside the browser tab.

---

## Architecture

```
gomoku/
├── index.html                          # UI + Canvas renderer + Pyodide bridge
├── game.py                             # Board class (make_move, undo_move, check_win)
├── ai1_minimax_anytimesearch.py        # Minimax + Alpha-Beta + Iterative Deepening
├── ai2_mcts.py                         # Pure MCTS with heuristic rollouts
└── ai3_minimax_anytimesearch_mcts.py   # Hybrid: MCTS exploration + Minimax refinement
```

At startup, Pyodide loads all `.py` files into the browser's virtual filesystem. On each AI turn, JavaScript passes the board state to Python, which runs the search and returns the chosen move.

---

## Evaluation Heuristic

All agents share the same `evaluate()` function: scans every 5-cell window across 4 directions and scores based on consecutive piece counts. A window with *k* pieces and no opponent pieces scores `SCORES[k]` where `{1:10, 2:100, 3:1000, 4:10000, 5:1000000}` — exponential scaling ensures the engine always prioritizes winning over building.

---

## Run Locally

```bash
git clone https://github.com/alexkujou/gomoku
cd gomoku
python -m http.server 8000
# open http://localhost:8000
```

Requires a local server (not `file://`) because Pyodide fetches `.py` files via HTTP.

---

## Author

**Alex Nguyen** · CS @ Arizona State University  
Research interests: Embodied AI, AI Planning, Human-Robot Interaction
