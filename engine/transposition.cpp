#include "transposition.h"
#include <random>

// ---------------------------------------------------------------------------
// ZOBRIST
// ---------------------------------------------------------------------------

ZobristTable::ZobristTable() {
    std::mt19937_64 rng(19022007);  // fixed seed -> reproducible
    for (int r = 0; r < kBoardSize; r++)
        for (int c = 0; c < kBoardSize; c++)
            for (int p = 0; p < 3; p++)
                table[r][c][p] = rng();
}

uint64_t ZobristTable::Hash(const Board& board) const {
    uint64_t hash = 0;
    for (int r = 0; r < kBoardSize; r++)
        for (int c = 0; c < kBoardSize; c++)
            if (board.grid[r][c] != 0)
                hash ^= table[r][c][board.grid[r][c]];
    return hash;
}

uint64_t ZobristTable::UpdateHash(uint64_t hash, int r, int c, int player) const {
    return hash ^ table[r][c][player];
}

// ---------------------------------------------------------------------------
// TRANSPOSITION TABLE
// ---------------------------------------------------------------------------

void TranspositionTable::Store(uint64_t hash, int depth, int score,
                                TTFlag flag, Move best_move) {
    auto it = table_.find(hash);
    // Only overwrite when new depth deeper
    if (it != table_.end() && it->second.depth > depth) return;
    table_[hash] = {depth, score, flag, best_move};
}

bool TranspositionTable::Lookup(uint64_t hash, int depth, int alpha, int beta,
                                 int& score, Move& best_move) const {
    auto it = table_.find(hash);
    if (it == table_.end()) return false;

    const TTEntry& entry = it->second;
    best_move = entry.best_move;

    if (entry.depth >= depth) {
        if (entry.flag == TTFlag::kExact) {
            score = entry.score; return true;
        }
        if (entry.flag == TTFlag::kLowerBound && entry.score >= beta) {
            score = entry.score; return true;
        }
        if (entry.flag == TTFlag::kUpperBound && entry.score <= alpha) {
            score = entry.score; return true;
        }
    }
    return false;
}

void TranspositionTable::Clear() {
    table_.clear();
}