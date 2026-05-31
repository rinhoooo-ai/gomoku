#include "evaluate.h"
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// CONSTANTS
// ---------------------------------------------------------------------------

struct Pattern {
    const char* str;
    int score;
};

// X = player, O = opponent, 0 = empty
constexpr Pattern kPatterns[] = {
    // Win
    {"XXXXX",   1'000'000},

    // Open-4
    {"0XXXX0",  50'000},

    // Half-4
    {"XXXX0",   10'000},
    {"0XXXX",   10'000},

    // Broken-4
    {"XX0XX",  8'000},  
    {"XXX0X",  8'000},
    {"X0XXX",  8'000},

    // Open-3
    {"0XXX0",   5'000},

    // Broken open-3
    {"0XX0X0",  4'000},
    {"0X0XX0",  4'000},

    // Half-3
    {"0XXX",    1'000},
    {"XXX0",    1'000},

    // Open-2
    {"0XX0",    500},

    // Half-2
    {"0XX",     100},
    {"XX0",     100},
};

// ---------------------------------------------------------------------------
// LINE EXTRACTION
// ---------------------------------------------------------------------------

static std::string LineToString(const Board& board, int sr, int sc, int dr, int dc, int player) {
    int opponent = 3 - player;
    std::string line;
    int r = sr, c = sc;
    while(board.InBounds(r, c)){
        if(board.grid[r][c] == player){
            line += 'X';
        } else {
            if(board.grid[r][c] == opponent){
                line += 'O';
            } else {
                line += '0';
            }
        }
        r += dr;
        c += dc;
    }
    return line;
}

static std::vector<std::string> ExtractLines(const Board& board, int player){
    std::vector<std::string> lines;

    // Rows
    for (int r = 0; r < kBoardSize; r++) {
        lines.push_back(LineToString(board, r, 0, 0, 1, player));
    }

    // Cols
    for (int c = 0; c < kBoardSize; c++) {
        lines.push_back(LineToString(board, 0, c, 1, 0, player));
    }

    // Diagonal top-left -> bottom-right
    for (int r = 0; r < kBoardSize; r++) {
        lines.push_back(LineToString(board, r, 0, 1, 1, player));
    }
    for (int c = 1; c < kBoardSize; c++) {
        lines.push_back(LineToString(board, 0, c, 1, 1, player));
    }

    // Diagonal top-right -> bottom-left
    for (int r = 0; r < kBoardSize; r++) {
        lines.push_back(LineToString(board, r, kBoardSize-1, 1, -1, player));
    }
    for (int c = 0; c < kBoardSize-1; c++) {
        lines.push_back(LineToString(board, 0, c, 1, -1, player));
    }

    return lines;
}

// ---------------------------------------------------------------------------
// SCORING
// ---------------------------------------------------------------------------

static int ScoreLine(const std::string& line){
    int score = 0;
    for(auto& p : kPatterns){
        size_t pos = 0;
        std::string s(p.str);
        while((pos = line.find(s, pos)) != std::string::npos){
            score += p.score;
            pos++;
        }
    }
    return score;
}

// ---------------------------------------------------------------------------
// EVALUATE
// ---------------------------------------------------------------------------

int Evaluate(const Board& board, int player){
    auto player_lines = ExtractLines(board, player);
    auto opponent_lines = ExtractLines(board, 3 - player);

    int score = 0;
    for(auto& line : player_lines){
        score += ScoreLine(line);
    }
    for(auto& line : opponent_lines){
        score -= ScoreLine(line);
    }
    return score;
}