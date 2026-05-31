#pragma once

#include "board.h"
#include "evaluate.h"
#include "transposition.h"
#include <vector>
#include <optional>

constexpr int kMaxDepth = 20;
constexpr int kCandidateRad = 2;
constexpr int kMaxCandidates = 8;

class Search {
public:
    Search();

    Move GetBestMove(Board& board, int player, double time_limit, std::optional<Move> last_move);

// private:
    ZobristTable zobrist_;
    TranspositionTable tt_;

    std::vector<Move> GetCandidates(const Board& board) const;
    std::vector<Move> GetSortedMoves(const Board& board, int player, int depth_remaining) const;

    int Minimax(Board& board, int depth, int alpha, int beta, bool is_maximizing, int player, 
                std::optional<Move> last_move, uint64_t hash, double start_time, double time_limit);

private:
    int QuickScore(const Board& board, int r, int c, int player) const;
};

