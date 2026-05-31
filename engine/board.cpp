#include "board.h"

bool Board::InBounds(int r, int c) const {
    return 0 <= r && r < kBoardSize && 0 <= c && c < kBoardSize;
}

bool Board::IsEmpty(int r, int c) const {
    return grid[r][c] == 0;
}

bool Board::IsFull() const {
    return move_count == kBoardSize * kBoardSize;
}

void Board::MakeMove(int r, int c, int player){
    grid[r][c] = player;
    move_count++;
}

void Board::UndoMove(int r, int c){
    grid[r][c] = 0;
    move_count--;
}

int Board::CountDir(int r, int c, int dr, int dc, int player) const {
    int cnt = 0;
    int nr = r + dr, nc = c + dc;

    while(InBounds(nr, nc) && grid[nr][nc] == player){
        cnt++;
        nr += dr;
        nc += dc;
    }
    return cnt;
}

bool Board::CheckWin(int r, int c, int player) const {
    constexpr int kDirs[4][2] = {{0,1},{1,0},{1,1},{1,-1}};
    for(auto& d : kDirs){
        int cnt = 1 + CountDir(r, c, d[0], d[1], player) + CountDir(r, c, -d[0], -d[1], player);

        if(cnt >= 5){
            return true;
        }
    }
    return false;
}