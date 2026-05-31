#include "engine/board.h"
#include "engine/search.h"
#include "engine/evaluate.h"
#include <emscripten/emscripten.h>
#include <vector>
#include <cstring>

// ---------------------------------------------------------------------------
// GLOBAL STATE
// ---------------------------------------------------------------------------

static Board   g_board;
static Search  g_search;
static int     g_move_history_r[kBoardSize * kBoardSize];
static int     g_move_history_c[kBoardSize * kBoardSize];
static int     g_move_history_p[kBoardSize * kBoardSize];
static int     g_history_count = 0;
static Move    g_last_move     = {-1, -1};

// ---------------------------------------------------------------------------
// EXPORTED FUNCTIONS
// ---------------------------------------------------------------------------

extern "C" {

EMSCRIPTEN_KEEPALIVE
void ResetBoard() {
    g_board       = Board{};
    g_history_count = 0;
    g_last_move     = {-1, -1};
}

EMSCRIPTEN_KEEPALIVE
bool MakeMove(int r, int c, int player) {
    if (!g_board.InBounds(r, c))       return false;
    if (!g_board.IsEmpty(r, c))        return false;
    if (g_board.IsFull())              return false;

    g_board.MakeMove(r, c, player);
    g_move_history_r[g_history_count] = r;
    g_move_history_c[g_history_count] = c;
    g_move_history_p[g_history_count] = player;
    g_history_count++;
    g_last_move = {r, c};
    return true;
}

EMSCRIPTEN_KEEPALIVE
bool CheckWin(int r, int c, int player) {
    return g_board.CheckWin(r, c, player);
}

EMSCRIPTEN_KEEPALIVE
bool IsFull() {
    return g_board.IsFull();
}

// Return flat array 15x15, so JS read from WASM memory directly
EMSCRIPTEN_KEEPALIVE
int* GetBoard() {
    static int flat[kBoardSize * kBoardSize];
    for (int r = 0; r < kBoardSize; r++)
        for (int c = 0; c < kBoardSize; c++)
            flat[r * kBoardSize + c] = g_board.grid[r][c];
    return flat;
}

// Return pointer of history arrays
EMSCRIPTEN_KEEPALIVE
int* GetHistoryR() { return g_move_history_r; }

EMSCRIPTEN_KEEPALIVE
int* GetHistoryC() { return g_move_history_c; }

EMSCRIPTEN_KEEPALIVE
int* GetHistoryP() { return g_move_history_p; }

EMSCRIPTEN_KEEPALIVE
int GetHistoryCount() { return g_history_count; }

// AI calculate moves then return r * 100 + c to let JS decode
EMSCRIPTEN_KEEPALIVE
int GetBestMove(int player, double time_limit) {
    std::optional<Move> last = (g_last_move.r == -1)
                             ? std::nullopt
                             : std::optional<Move>{g_last_move};
    Move m = g_search.GetBestMove(g_board, player, time_limit, last);
    return m.r * 100 + m.c;
}

} // extern "C"