#include "engine/board.h"
#include "engine/evaluate.h"
#include "engine/search.h"
#include <cassert>
#include <cstdio>

// ---------------------------------------------------------------------------
// HELPER
// ---------------------------------------------------------------------------

static void ClearBoard(Board& board) {
    for (int r = 0; r < kBoardSize; r++)
        for (int c = 0; c < kBoardSize; c++)
            board.grid[r][c] = 0;
    board.move_count = 0;
}

static void Pass(const char* name) {
    printf("PASS: %s\n", name);
}

// ---------------------------------------------------------------------------
// TESTS
// ---------------------------------------------------------------------------

void TestCheckWin() {
    // 5 in a row horizontal
    {
        Board board;
        for (int c = 0; c < 5; c++) board.MakeMove(7, c, 1);
        assert(board.CheckWin(7, 4, 1));
    }

    // 5 in a row vertical
    {
        Board board;
        for (int r = 0; r < 5; r++) board.MakeMove(r, 7, 2);
        assert(board.CheckWin(4, 7, 2));
    }

    // 5 in a row diagonal
    {
        Board board;
        for (int i = 0; i < 5; i++) board.MakeMove(i, i, 1);
        assert(board.CheckWin(4, 4, 1));
    }

    // Not win yet
    {
        Board board;
        for (int c = 0; c < 4; c++) board.MakeMove(7, c, 1);
        assert(!board.CheckWin(7, 3, 1));
    }

    Pass("CheckWin");
}

void TestEvaluate() {
    Board board;

    // Empty board = 0
    assert(Evaluate(board, 1) == 0);

    // Open-3 should score higher than half-3
    board.MakeMove(7, 5, 1);
    board.MakeMove(7, 6, 1);
    board.MakeMove(7, 7, 1);
    int open3_score = Evaluate(board, 1);
    ClearBoard(board);

    board.MakeMove(7, 5, 1);
    board.MakeMove(7, 6, 1);
    board.MakeMove(7, 7, 1);
    board.MakeMove(7, 4, 2);  // block one end
    int half3_score = Evaluate(board, 1);
    ClearBoard(board);

    assert(open3_score > half3_score);

    Pass("Evaluate");
}

void TestGetBestMove() {
    Board  board;
    Search search;

    // Take win: 4 in a row → play 5th
    for (int c = 0; c < 4; c++) board.MakeMove(7, c, 1);
    Move m = search.GetBestMove(board, 1, 5.0, std::nullopt);
    assert(m.r == 7 && m.c == 4);
    ClearBoard(board);
    Pass("GetBestMove: takes win");

    // Block opponent 4 in a row
    for (int c = 0; c < 4; c++) board.MakeMove(7, c, 2);
    m = search.GetBestMove(board, 1, 5.0, Move{7, 3});
    assert(m.r == 7 && (m.c == 4 || m.c == 0));
    ClearBoard(board);
    Pass("GetBestMove: blocks opponent 4");

    // Block open-3
    board.MakeMove(7, 6, 2);
    board.MakeMove(7, 7, 2);
    board.MakeMove(7, 8, 2);
    m = search.GetBestMove(board, 1, 5.0, Move{7, 8});
    assert(m.r == 7 && (m.c == 5 || m.c == 9));
    ClearBoard(board);
    Pass("GetBestMove: blocks open-3");
}

// ---------------------------------------------------------------------------
// MAIN
// ---------------------------------------------------------------------------

int main() {
    printf("Running tests...\n\n");

    TestCheckWin();
    TestEvaluate();
    TestGetBestMove();

    printf("\n=== All tests passed! ===\n");
    return 0;
}