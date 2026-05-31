#pragma once

#include "board.h"

constexpr int kWinScore  =  1'000'000;
constexpr int kLoseScore = -1'000'000;

int Evaluate(const Board& board, int player);