# Gomoku Engine

A Gomoku (5-in-a-row, 15x15) AI engine written in C++17, compiled to WebAssembly to run directly in the browser. No dependencies beyond Emscripten.

## Demo

Open `web/index.html` (prebuilt `gomoku.js` / `gomoku.wasm` included) via a static server:

```bash
cd web
python3 -m http.server 8000
# open http://localhost:8000
```

## Engine architecture

```
engine/
├── board.{h,cpp}          # 15x15 board, place/undo move, win check (5 directions)
├── evaluate.{h,cpp}        # heuristic scoring via pattern-matching on lines
├── transposition.{h,cpp}   # Zobrist hashing + transposition table (unordered_map)
└── search.{h,cpp}          # Minimax + alpha-beta, iterative deepening
```

### Search (`search.cpp`)
- **Iterative deepening** up to `kMaxDepth = 20`, cut off by the `time_limit` (seconds) passed to `GetBestMove`.
- **Minimax with alpha-beta pruning**, backed by a transposition table (Zobrist hash) caching results per `(hash, depth)`.
- **Candidate generation** is restricted to a radius (`kCandidateRad = 2`) around existing stones instead of scanning the full board — avoids a branching factor of 225.
- **Move ordering**: an immediate winning move or a move blocking the opponent's win is given absolute priority before sorting the rest by `QuickScore` (counts consecutive stones in 4 directions, squared).
- The number of candidates examined per node shrinks with remaining depth (`kMaxCandidates = 8`, down to 5–7 near the leaves) to keep search fast.

### Evaluate (`evaluate.cpp`)
Extracts every row/column/diagonal into a string (`X` = self, `O` = opponent, `0` = empty), then matches it against a fixed pattern table (five, open-4, half-4, broken-4, open-3, open-2, half-2, etc.) to add/subtract score. Pure heuristic — no neural net.

### Transposition table
Standard Zobrist hashing (`table[r][c][player]`), storing `depth / score / flag (exact, lower, upper bound) / best_move`, used both for pruning and move ordering (the TT move is bubbled to the front of the candidate list).

## Build

Requires [Emscripten](https://emscripten.org/docs/getting_started/downloads.html) (`em++`) on your `PATH`.

```bash
# build WASM for the web target (outputs web/gomoku.js + web/gomoku.wasm)
make build

# run the test suite (plain g++, no Emscripten needed)
make test

# clean WASM build output
make clean
```

`main.cpp` exports the following C functions via Emscripten for JS to call directly:

| Function | Description |
|---|---|
| `ResetBoard()` | resets the board and move history |
| `MakeMove(r, c, player)` | places a stone, returns `false` if invalid |
| `CheckWin(r, c, player)` | checks for a win at the last-played position |
| `IsFull()` | board full → draw |
| `GetBoard()` | returns a pointer to a flat 15x15 array (read directly from WASM memory) |
| `GetHistoryR/C/P()`, `GetHistoryCount()` | move history |
| `GetBestMove(player, time_limit)` | AI computes the best move, returns `r*100+c` |

## Web UI

`web/main.js` loads the WASM module, draws the board with Canvas 2D, and reads board state directly from `Module.HEAP32` (no JSON round-trip). The player picks black/white on the setup screen; the AI moves automatically on its turn (default time limit: 15s/move).
