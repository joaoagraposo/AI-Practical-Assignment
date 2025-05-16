import random
import csv
from connect4 import ConnectFourState
from connect4 import pure_monte_carlo_choice
from tqdm import tqdm
from copy import deepcopy

# (Reutiliza a sua classe e função existentes)
# from your_module import ConnectFourState, pure_monte_carlo_choice

def state_to_feature_vector(state):
    """
    Transforma um estado de jogo num vetor de features:
    - flatten(board) com valores em {-1,0,1}
    - jogador a mover (1 ou -1)
    """
    features = []
    for row in state.board:
        features.extend(row)
    features.append(state.player)
    return features

def generate_connect_four_dataset(num_samples: int,
                                  playouts: int,
                                  filename: str):
    """
    Gera um dataset de (estado, melhor_jogada) para Connect-Four.
    
    Para cada amostra:
      1. Parte de tabuleiro vazio
      2. Avança um número aleatório de jogadas randómicas (até 20) para diversificar estados
      3. Se o estado não for terminal, usa pure_monte_carlo_choice para obter a melhor jogada
      4. Extrai features e grava (features..., melhor_jogada)
    """
    header = [f"cell_{i}" for i in range(ConnectFourState.ROWS * ConnectFourState.COLS)] + ["player", "move"]
    
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for _ in tqdm(range(num_samples), desc="Gerar dataset"):
            state = ConnectFourState()
            
            # 1. Avança por um número aleatório de jogadas
            depth = random.randint(0, 20)
            for _ in range(depth):
                legal = state.get_legal_moves()
                if not legal or state.check_win() is not None:
                    break
                state.play_move(random.choice(legal))
            
            # 2. Ignora estados terminais
            if state.check_win() is not None:
                continue
            
            # 3. Calcula melhor jogada com Monte Carlo (substituir por MCTS/UCT se quiser)
            best_move = pure_monte_carlo_choice(state, playouts)
            
            # 4. Extrai vetor de features e escreve linha (features + label)
            features = state_to_feature_vector(state)
            writer.writerow(features + [best_move])
    
    print(f"Dataset guardado em: {filename}")

if __name__ == "__main__":
    # Exemplo: 5000 amostras, 200 simulações por jogada
    generate_connect_four_dataset(num_samples=5000,
                                  playouts=200,
                                  filename="connect4_dataset.csv")
