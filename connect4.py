import random
import time
from copy import deepcopy

# Estado do jogo Connect-Four
class ConnectFourState:
    ROWS = 6
    COLS = 7

    def __init__(self, board=None, player=1):
        # 0 = vazio, 1 = X, -1 = O
        self.board = deepcopy(board) if board is not None else [[0]*self.COLS for _ in range(self.ROWS)]
        self.player = player

    def clone(self):
        return ConnectFourState(self.board, self.player)

    def get_legal_moves(self):
        return [c for c in range(self.COLS) if self.board[0][c] == 0]

    def play_move(self, col):
        '''Executa jogada na coluna indicada; retorna True se válida.'''
        for r in range(self.ROWS-1, -1, -1):
            if self.board[r][col] == 0:
                self.board[r][col] = self.player
                self.player *= -1
                return True
        return False

    def check_win(self):
        '''Deteta vitória (1/-1), empate (0) ou None se não terminado.'''
        directions = [(0,1),(1,0),(1,1),(1,-1)]
        for r in range(self.ROWS):
            for c in range(self.COLS):
                p = self.board[r][c]
                if p == 0:
                    continue
                for dr, dc in directions:
                    cnt, rr, cc = 0, r, c
                    for _ in range(4):
                        if 0 <= rr < self.ROWS and 0 <= cc < self.COLS and self.board[rr][cc] == p:
                            cnt += 1; rr += dr; cc += dc
                        else:
                            break
                    if cnt == 4:
                        return p
        if not self.get_legal_moves():
            return 0
        return None

# Função para imprimir o tabuleiro
def print_board(state):
    print('\n  ' + ' '.join(str(c) for c in range(state.COLS)))
    for row in state.board:
        print(' |' + ' '.join('X' if v==1 else 'O' if v==-1 else '.' for v in row) + '|')
    print()

# Pure Monte Carlo Game Search
def pure_monte_carlo_choice(state, playouts=1000):
    '''Para cada movimento legal, executa playouts aleatórios e escolhe o movimento com mais vitórias.'''
    best_move, best_wins = None, -1
    for mv in state.get_legal_moves():
        wins = 0
        for _ in range(playouts):
            sim = state.clone()
            sim.play_move(mv)
            while sim.check_win() is None:
                sim.play_move(random.choice(sim.get_legal_moves()))
            if sim.check_win() == state.player:
                wins += 1
        if wins > best_wins:
            best_wins, best_move = wins, mv
    return best_move

# Leitura de jogada humana
def human_move(state):
    moves = state.get_legal_moves()
    while True:
        try:
            col = int(input(f"Jogador {'1 (X)' if state.player==1 else '2 (O)'}, escolha coluna {moves}: "))
            if col in moves:
                return col
        except ValueError:
            pass
        print("Movimento inválido.")

# Loop principal do jogo
def play_game():
    print("=== Connect-Four com Pure Monte Carlo ===")
    # escolha de modo de jogo
    while True:
        mode = input("Modo: 1) PvP  2) PvC\nEscolha: ")
        if mode in ('1','2'):
            break
    playouts = 1000  # número de simulações por movimento
    state = ConnectFourState()
    print_board(state)
    while True:
        if mode == '1' or state.player == 1:
            mv = human_move(state)
        else:
            print("Computador a pensar...")
            start = time.time()
            mv = pure_monte_carlo_choice(state, playouts)
            print(f"Tempo de cálculo: {time.time() - start:.2f}s")
        state.play_move(mv)
        print_board(state)
        result = state.check_win()
        if result is not None:
            if result == 0:
                print("Empate!")
            else:
                winner = '1 (X)' if result == 1 else '2 (O)'
                print(f"Vitória do jogador {winner}!")
            break

if __name__ == '__main__':
    play_game()
