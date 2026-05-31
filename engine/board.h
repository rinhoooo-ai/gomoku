#pragma once

constexpr int kBoardSize = 15;

struct Move {
    int r, c;
};

class Board {
public:
    int grid[kBoardSize][kBoardSize]{};
    int move_count = 0;

    bool InBounds(int r, int c) const;
    bool IsEmpty(int r, int c)  const;
    bool IsFull()               const;

    void MakeMove(int r, int c, int player);
    void UndoMove(int r, int c);

    bool CheckWin(int r, int c, int player) const;
private:
    int CountDir(int r, int c, int dr, int dc, int player) const;
};