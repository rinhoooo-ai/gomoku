#pragma once

#include "board.h"
#include <cstdint>
#include <unordered_map>

// ---------------------------------------------------------------------------
// ZOBRIST HASHING
// ---------------------------------------------------------------------------

class ZobristTable {
public:
    ZobristTable();
    uint64_t Hash(const Board& board) const;
    uint64_t UpdateHash(uint64_t hash, int r, int c, int player) const;

private:
    // table[r][c][player]: player = 1 or 2
    uint64_t table[kBoardSize][kBoardSize][3];
};

// ---------------------------------------------------------------------------
// TRANSPOSITION TABLE
// ---------------------------------------------------------------------------

enum class TTFlag {
    kExact,      // exact score
    kLowerBound, // alpha cutoff when score >= value
    kUpperBound, // beta cutoff  when score <= value
};

struct TTEntry {
    int      depth;
    int      score;
    TTFlag   flag;
    Move     best_move;
};

class TranspositionTable {
public:
    void Store(uint64_t hash, int depth, int score, TTFlag flag, Move best_move);
    bool Lookup(uint64_t hash, int depth, int alpha, int beta,
                int& score, Move& best_move) const;
    void Clear();

private:
    std::unordered_map<uint64_t, TTEntry> table_;
};