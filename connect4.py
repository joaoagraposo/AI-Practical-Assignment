import random
import time
from copy import deepcopy

import pandas as pd
import numpy as np

from DeciTree import buildTree, classifyExample

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

# Impressão do tabuleiro
def print_board(state):
    print('\n  ' + ' '.join(str(c) for c in range(state.COLS)))
    for row in state.board:
        print(' |' + ' '.join('X' if v==1 else 'O' if v==-1 else '.' for v in row) + '|')
    print()

# Pure Monte Carlo Search
def pure_monte_carlo_choice(state, playouts=1000):
    best_move, best_wins = None, -1
    for mv in state.get_legal_moves():
        wins = 0
        for _ in range(playouts):
            sim = state.clone()
            sim.play_move(mv)
            while sim.check_win() is None:
                sim.play_move(random.choice(sim.get_legal_moves()))
            # ganha quem jogou mv se sim.player foi invertido
            if sim.check_win() == state.player:
                wins += 1
        if wins > best_wins:
            best_wins, best_move = wins, mv
    return best_move

# Flatten do estado para classificação
def state_to_series(state, feature_columns):
    vec = []
    for row in state.board:
        vec.extend(row)
    vec.append(state.player)
    return pd.Series(vec, index=feature_columns)

# Loop principal
def play_game():
    print("=== Connect-Four com MCTS e ID3 ===")
    print("Modo: 1) PvP  2) PvC MCTS  3) C-ID3 vs C-MCTS")
    mode = input("Escolha: ").strip()
    while mode not in ('1','2','3'):
        mode = input("Escolha: ").strip()

    # para modo 3, treinamos a árvore sem discretizar
    if mode == '3':
        print("\nTreinando árvore de decisão (ID3)...")
        df = pd.read_csv("connect4_dataset.csv")
        # não discretizar com quartis para manter labels inteiros
        feature_columns = df.columns[:-1]
        X = df[feature_columns]
        y = df.iloc[:, -1]
        max_depth = int(np.log2(len(df)) + 1)
        training_data = pd.concat([X, y], axis=1)
        decision_tree = buildTree(training_data, max_depth)
        print("Árvore treinada!\n")

    state = ConnectFourState()
    print_board(state)

    while True:
        if mode == '1' or (mode == '2' and state.player == 1):
            # humano
            col = int(input(f"Jogador {'1 (X)' if state.player==1 else '2 (O)'}, escolha coluna {state.get_legal_moves()}: "))

        elif mode == '2' and state.player == -1:
            # PvC MCTS
            print("Computador (MCTS) a pensar...")
            start = time.time()
            col = pure_monte_carlo_choice(state, playouts=1000)
            print(f"Tempo de cálculo: {time.time() - start:.2f}s")

        else:
            # modo 3
            if state.player == 1:
                print("Computador (ID3) a pensar...")
                example = state_to_series(state, feature_columns)
                # classifyExample retorna já inteiro
                col = int(classifyExample(example, decision_tree))
            else:
                print("Computador (MCTS) a pensar...")
                start = time.time()
                col = pure_monte_carlo_choice(state, playouts=1000)
                print(f"Tempo de cálculo: {time.time() - start:.2f}s")

        state.play_move(col)
        print_board(state)
        res = state.check_win()
        if res is not None:
            if res == 0:
                print("Empate!")
            else:
                w = '1 (X)' if res == 1 else '2 (O)'
                print(f"Vitória do jogador {w}!")
            break

if __name__ == '__main__':
    play_game()
