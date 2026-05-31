#include "search.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <climits>
#include <cstdio>

// ---------------------------------------------------------------------------
// TIME HELPER
// ---------------------------------------------------------------------------

static double Now(){
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();    
}

// ---------------------------------------------------------------------------
// CONSTRUCTOR
// ---------------------------------------------------------------------------

Search::Search() : zobrist_(), tt_() {}

// ---------------------------------------------------------------------------
// CANDIDATE MOVES
// ---------------------------------------------------------------------------

std::vector<Move> Search::GetCandidates(const Board& board) const {
    bool found = false;
    for(int r = 0; r < kBoardSize; r++){
        for(int c = 0; c < kBoardSize; c++){
            if(board.grid[r][c] != 0){
                found = true;
                break;
            }
        }
    }

    if(!found){
        return {{kBoardSize / 2, kBoardSize / 2}};
    }

    std::vector<Move> candidates;
    bool seen[kBoardSize][kBoardSize]{};

    for(int r = 0; r < kBoardSize; r++){
        for(int c = 0; c < kBoardSize; c++){
            if(board.grid[r][c] == 0){
                continue;
            }

            for(int dr = -kCandidateRad; dr <= kCandidateRad; dr++){
                for(int dc = -kCandidateRad; dc <= kCandidateRad; dc++){
                    int nr = r + dr;
                    int nc = c + dc;

                    if(!board.InBounds(nr, nc)){
                        continue;
                    }
                    if(board.grid[nr][nc] != 0){
                        continue;
                    }
                    if(seen[nr][nc]){
                        continue;
                    }
                    seen[nr][nc] = true;
                    candidates.push_back({nr, nc});
                }
            }
        }
    }
    return candidates;
}

int Search::QuickScore(const Board& board, int r, int c, int player) const {
    int score    = 0;
    int opponent = 3 - player;
    constexpr int kDirs[4][2] = {{0,1},{1,0},{1,1},{1,-1}};

    for (auto& d : kDirs) {
        int cnt_p = 1;
        for (int sign : {1, -1}) {
            int nr = r + sign*d[0], nc = c + sign*d[1];
            while (board.InBounds(nr, nc) && board.grid[nr][nc] == player) {
                cnt_p++;
                nr += sign*d[0];
                nc += sign*d[1];
            }
        }

        int cnt_o = 1;
        for (int sign : {1, -1}) {
            int nr = r + sign*d[0], nc = c + sign*d[1];
            while (board.InBounds(nr, nc) && board.grid[nr][nc] == opponent) {
                cnt_o++;
                nr += sign*d[0];
                nc += sign*d[1];
            }
        }

        score += cnt_p * cnt_p * 10;
        score += cnt_o * cnt_o * 8;
    }
    return score;
}

std::vector<Move> Search::GetSortedMoves(const Board& board, int player,
                                          int depth_remaining) const {
    int max_cand = depth_remaining <= 1 ? 5
                 : depth_remaining <= 2 ? 7
                 : kMaxCandidates;

    auto candidates = GetCandidates(board);

    std::vector<std::pair<int, Move>> scored;
    for (auto& m : candidates) {
        // Check win ngay — ưu tiên tuyệt đối
        const_cast<Board&>(board).MakeMove(m.r, m.c, player);
        bool wins = board.CheckWin(m.r, m.c, player);
        const_cast<Board&>(board).UndoMove(m.r, m.c);
        if (wins) return {m};

        // Check block opponent win
        int opponent = 3 - player;
        const_cast<Board&>(board).MakeMove(m.r, m.c, opponent);
        bool blocks = board.CheckWin(m.r, m.c, opponent);
        const_cast<Board&>(board).UndoMove(m.r, m.c);

        int s = QuickScore(board, m.r, m.c, player);
        if (blocks) s += 100000;  // ưu tiên cao nhưng vẫn sort

        scored.push_back({s, m});
    }

    std::sort(scored.begin(), scored.end(),
              [](auto& a, auto& b){ return a.first > b.first; });

    std::vector<Move> result;
    for (int i = 0; i < std::min((int)scored.size(), max_cand); i++)
        result.push_back(scored[i].second);
    return result;
}

// ---------------------------------------------------------------------------
// MINIMAX
// ---------------------------------------------------------------------------

int Search::Minimax(Board& board, int depth, int alpha, int beta, bool is_maximizing, int player, 
                    std::optional<Move> last_move, uint64_t hash, double start_time, double time_limit){
    // Time check
    if(Now() - start_time >= time_limit){
        return INT_MIN;
    }

    // Terminal: win check
    if(last_move.has_value()){
        int last_player = is_maximizing ? 3 - player : player;
        if(board.CheckWin(last_move->r, last_move->c, last_player)){
            return last_player == player ? kWinScore : kLoseScore;
        }
    }

    // Terminal: depth or full board
    if(depth == 0 || board.IsFull()){
        return Evaluate(board, player);
    }

    // Transposition table lookup
    Move tt_move = {-1, -1};
    int  tt_score;
    if(tt_.Lookup(hash, depth, alpha, beta, tt_score, tt_move)){
        return tt_score;
    }

    int cur_player = is_maximizing ? player : 3 - player;
    auto moves = GetSortedMoves(board, cur_player, depth);

    // TT best move on top
    if(tt_move.r != -1){
        auto it = std::find_if(moves.begin(), moves.end(), [&](const Move& m){ return m.r == tt_move.r && m.c == tt_move.c; });

        if(it != moves.end()){
            std::rotate(moves.begin(), it, it + 1);
        }
    }

    Move best_move = moves.empty() ? Move{-1,-1} : moves[0];
    TTFlag flag = TTFlag::kUpperBound;
    int best = is_maximizing ? INT_MIN : INT_MAX;

    for(auto& m : moves){
        if(Now() - start_time >= time_limit){
            return INT_MIN;
        }

        uint64_t new_hash = zobrist_.UpdateHash(hash, m.r, m.c, cur_player);
        board.MakeMove(m.r, m.c, cur_player);
        int score = Minimax(board, depth - 1, alpha, beta, !is_maximizing, player, m, new_hash, start_time, time_limit);
        board.UndoMove(m.r, m.c);

        if(score == INT_MIN){
            return INT_MIN; // time out
        }

        if(is_maximizing){
            if(score > best){
                best = score;
                best_move = m;
            }

            alpha = std::max(alpha, best);

            if(beta <= alpha){
                flag = TTFlag::kLowerBound;
                break;
            }
        } else {
            if(score < best){
                best = score;
                best_move = m;
            }

            beta = std::min(beta, best);

            if(beta <= alpha){
                flag = TTFlag::kUpperBound;
                break;
            }
        }

        if(flag == TTFlag::kUpperBound && best > INT_MIN){
            flag = TTFlag::kExact;
        }
    }

    tt_.Store(hash, depth, best, flag, best_move);
    return best;
}

// ---------------------------------------------------------------------------
// GET BEST MOVE — iterative deepening
// ---------------------------------------------------------------------------

Move Search::GetBestMove(Board& board, int player, double time_limit, std::optional<Move> last_move) {
    double start = Now();
    tt_.Clear();

    uint64_t hash = zobrist_.Hash(board);

    // Fallback
    auto candidates = GetSortedMoves(board, player, kMaxCandidates);
    Move best_move  = candidates.empty() ? Move{kBoardSize/2, kBoardSize/2} : candidates[0];

    for(int depth = 1; depth <= kMaxDepth; depth++){
        if(Now() - start >= time_limit){
            break;
        }

        Move  candidate_move  = {-1, -1};
        int   candidate_score = INT_MIN;
        bool  timed_out       = false;

        auto moves = GetSortedMoves(board, player, depth);
        for(auto& m : moves){
            if(Now() - start >= time_limit){
                timed_out = true;
                break;
            }

            uint64_t new_hash = zobrist_.UpdateHash(hash, m.r, m.c, player);
            board.MakeMove(m.r, m.c, player);
            int score = Minimax(board, depth - 1, INT_MIN, INT_MAX, false, player, m, new_hash, start, time_limit);
            board.UndoMove(m.r, m.c);

            if(score == INT_MIN){
                timed_out = true;
                break;
            }

            if(score > candidate_score){
                candidate_score = score;
                candidate_move = m;
            }

            if(candidate_score >= kWinScore){
                break;
            }
        }

        if(!timed_out && candidate_move.r != -1){
            best_move = candidate_move;

            printf("[Search] depth=%d score=%d time=%.2fs\n", depth, candidate_score, Now() - start);
        }

        if(timed_out || candidate_score >= kWinScore){
            break;
        }
    }

    return best_move;
}