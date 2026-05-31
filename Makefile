CXX      = em++
CXXFLAGS = -O2 -std=c++17

SRCS = engine/board.cpp \
       engine/evaluate.cpp \
       engine/transposition.cpp \
       engine/search.cpp \
       main.cpp

OUT = web/gomoku.js

EXPORTS = "_ResetBoard,_MakeMove,_CheckWin,_IsFull,_GetBoard,\
_GetHistoryR,_GetHistoryC,_GetHistoryP,_GetHistoryCount,_GetBestMove"

build:
	$(CXX) $(CXXFLAGS) $(SRCS) \
		-s WASM=1 \
		-s EXPORTED_FUNCTIONS=$(EXPORTS) \
		-s EXPORTED_RUNTIME_METHODS='["HEAP32"]' \
		-s ALLOW_MEMORY_GROWTH=1 \
		-s MODULARIZE=0 \
		-o $(OUT)

clean:
	rm -f web/gomoku.js web/gomoku.wasm


test:
	g++ -O2 -std=c++17 \
		engine/board.cpp \
		engine/evaluate.cpp \
		engine/transposition.cpp \
		engine/search.cpp \
		test.cpp \
		-o test_runner.exe
	./test_runner.exe