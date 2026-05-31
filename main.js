Module.onRuntimeInitialized = () => {

// ---------------------------------------------------------------------------
// CONSTANTS
// ---------------------------------------------------------------------------

const BOARD_SIZE   = 15;
const CELL_SIZE    = 40;
const PADDING      = 20;
const STONE_RADIUS = 16;

// ---------------------------------------------------------------------------
// WASM FUNCTIONS
// ---------------------------------------------------------------------------

const ResetBoard      = () => Module._ResetBoard();
const MakeMove        = (r, c, p) => Module._MakeMove(r, c, p);
const CheckWin        = (r, c, p) => Module._CheckWin(r, c, p);
const IsFull          = () => Module._IsFull();
const GetBoard        = () => Module._GetBoard();
const GetHistoryR     = () => Module._GetHistoryR();
const GetHistoryC     = () => Module._GetHistoryC();
const GetHistoryP     = () => Module._GetHistoryP();
const GetHistoryCount = () => Module._GetHistoryCount();
const GetBestMove     = (p, t) => Module._GetBestMove(p, t);

// ---------------------------------------------------------------------------
// STATE
// ---------------------------------------------------------------------------

let playerSide   = 1;
let aiSide       = 2;
let isPlayerTurn = true;
let gameOver     = false;
let isThinking   = false;

// ---------------------------------------------------------------------------
// CANVAS
// ---------------------------------------------------------------------------

const canvas = document.getElementById('board-canvas');
const ctx    = canvas.getContext('2d');

function DrawBoard() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Background
    ctx.fillStyle = '#1e1a14';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Grid lines
    ctx.strokeStyle = '#5a4a32';
    ctx.lineWidth   = 1;

    for (let i = 0; i < BOARD_SIZE; i++) {
        const x = PADDING + i * CELL_SIZE;
        const y = PADDING + i * CELL_SIZE;

        ctx.beginPath();
        ctx.moveTo(x, PADDING);
        ctx.lineTo(x, PADDING + (BOARD_SIZE - 1) * CELL_SIZE);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(PADDING, y);
        ctx.lineTo(PADDING + (BOARD_SIZE - 1) * CELL_SIZE, y);
        ctx.stroke();
    }

    // Star points
    const stars = [[7,7],[3,3],[3,11],[11,3],[11,11]];
    ctx.fillStyle = '#5a4a32';
    for (const [r, c] of stars) {
        ctx.beginPath();
        ctx.arc(PADDING + c * CELL_SIZE, PADDING + r * CELL_SIZE, 3, 0, Math.PI * 2);
        ctx.fill();
    }

    // Stones
    const ptr   = GetBoard();
    const board = new Int32Array(Module.HEAP32.buffer, ptr, BOARD_SIZE * BOARD_SIZE);

    for (let r = 0; r < BOARD_SIZE; r++) {
        for (let c = 0; c < BOARD_SIZE; c++) {
            const cell = board[r * BOARD_SIZE + c];
            if (cell === 0) continue;

            const x = PADDING + c * CELL_SIZE;
            const y = PADDING + r * CELL_SIZE;

            const grad = ctx.createRadialGradient(x - 4, y - 4, 2, x, y, STONE_RADIUS);
            if (cell === 1) {
                grad.addColorStop(0, '#666');
                grad.addColorStop(1, '#111');
            } else {
                grad.addColorStop(0, '#ffffff');
                grad.addColorStop(1, '#aaaaaa');
            }

            ctx.beginPath();
            ctx.arc(x, y, STONE_RADIUS, 0, Math.PI * 2);
            ctx.fillStyle = grad;
            ctx.fill();
        }
    }
}

// ---------------------------------------------------------------------------
// HISTORY
// ---------------------------------------------------------------------------

const COLS = 'ABCDEFGHIJKLMNO';

function UpdateHistory() {
    const count = GetHistoryCount();
    const rs    = new Int32Array(Module.HEAP32.buffer, GetHistoryR(), count);
    const cs    = new Int32Array(Module.HEAP32.buffer, GetHistoryC(), count);
    const ps    = new Int32Array(Module.HEAP32.buffer, GetHistoryP(), count);

    const list  = document.getElementById('history-list');
    list.innerHTML = '';

    for (let i = 0; i < count; i++) {
        const div     = document.createElement('div');
        div.className = `history-item ${ps[i] === 1 ? 'black' : 'white'}`;
        const label   = ps[i] === 1 ? '⚫' : '⚪';
        div.textContent = `${i + 1}. ${label} ${COLS[cs[i]]}${BOARD_SIZE - rs[i]}`;
        list.appendChild(div);
    }

    list.scrollTop = list.scrollHeight;
}

// ---------------------------------------------------------------------------
// STATUS
// ---------------------------------------------------------------------------

function SetStatus(text) {
    document.getElementById('status-text').textContent = text;
}

// ---------------------------------------------------------------------------
// WIN CHECK
// ---------------------------------------------------------------------------

function CheckWinAndEnd(r, c, player) {
    if (!CheckWin(r, c, player)) return false;

    gameOver = true;
    const winner = player === playerSide ? 'You win! 🎉' : 'AI wins!';
    SetStatus(winner);

    ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle    = '#ffffff';
    ctx.font         = '28px Segoe UI';
    ctx.textAlign    = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(winner, canvas.width / 2, canvas.height / 2);

    return true;
}

// ---------------------------------------------------------------------------
// AI MOVE
// ---------------------------------------------------------------------------

async function DoAiMove() {
    if (gameOver) return;

    isThinking = true;
    SetStatus('AI is thinking...');
    await new Promise(r => setTimeout(r, 10));

    const encoded = GetBestMove(aiSide, 15.0);
    const r       = Math.floor(encoded / 100);
    const c       = encoded % 100;

    MakeMove(r, c, aiSide);
    DrawBoard();
    UpdateHistory();

    isThinking = false;

    if (!CheckWinAndEnd(r, c, aiSide)) {
        if (IsFull()) {
            SetStatus('Draw!');
            gameOver = true;
        } else {
            isPlayerTurn = true;
            SetStatus('Your turn');
        }
    }
}

// ---------------------------------------------------------------------------
// PLAYER CLICK
// ---------------------------------------------------------------------------

canvas.addEventListener('click', async (e) => {
    if (!isPlayerTurn || gameOver || isThinking) return;

    const rect = canvas.getBoundingClientRect();
    const x    = e.clientX - rect.left;
    const y    = e.clientY - rect.top;

    const c = Math.round((x - PADDING) / CELL_SIZE);
    const r = Math.round((y - PADDING) / CELL_SIZE);

    if (c < 0 || c >= BOARD_SIZE || r < 0 || r >= BOARD_SIZE) return;

    const ok = MakeMove(r, c, playerSide);
    if (!ok) return;

    isPlayerTurn = false;
    DrawBoard();
    UpdateHistory();

    if (!CheckWinAndEnd(r, c, playerSide)) {
        if (IsFull()) {
            SetStatus('Draw!');
            gameOver = true;
        } else {
            await DoAiMove();
        }
    }
});

// ---------------------------------------------------------------------------
// GAME FLOW
// ---------------------------------------------------------------------------

window.startGame = (side) => {
    playerSide = side;
    aiSide     = 3 - side;
    gameOver   = false;
    isThinking = false;

    document.getElementById('setup-screen').classList.add('hidden');
    document.getElementById('game-screen').classList.remove('hidden');

    ResetBoard();
    DrawBoard();
    UpdateHistory();

    if (playerSide === 2) {
        isPlayerTurn = false;
        DoAiMove();
    } else {
        isPlayerTurn = true;
        SetStatus('Your turn');
    }
};

window.newGame = () => {
    gameOver     = false;
    isThinking   = false;
    isPlayerTurn = true;
    document.getElementById('setup-screen').classList.remove('hidden');
    document.getElementById('game-screen').classList.add('hidden');
};

// WASM ready
console.log('WASM loaded');

}; // end onRuntimeInitialized