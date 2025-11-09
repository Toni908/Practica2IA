""" 
Segundo intentop, esta vez con SARSA.

Autor: Antonio Garcia Font
"""
import time
import gymnasium as gym
import numpy as np
import random

def choose_action(Q, state, epsilon, env):
    """Política epsilon-greedy"""
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])
    
def render_policy(env, policy, delay=0.4):
    state, _ = env.reset()
    done = False

    while not done:
        action = policy[state]
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        time.sleep(delay)

    print(f"Recompensa final: {reward}")

def main():
    # Crear entorno FrozenLake
    env = gym.make("FrozenLake-v1", is_slippery=True, map_name="4x4", render_mode='ansi')
    
    gamma = 0.99
    alpha = 0.1
    epsilon = 0.1
    episodes = 50000
    
    # Inicializar Q(s,a)
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        first = True    # <-- diferencia con Q-Learning

        while not done:
            # Elegir acción
            if first:   # <-- diferencia con Q-Learning
                action = choose_action(Q, state, epsilon, env)  
                first = False       # <-- diferencia con Q-Learning
            
            # Ejecutar acción
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Q-Learning update
            next_action = choose_action(Q, next_state, epsilon, env)    # <-- diferencia con Q-Learning
            Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])

            state = next_state
            action = next_action   # <-- diferencia con Q-Learning
        
        if ep % 2000 == 0:
            print(f"Episodio {ep} | Q parcial:")
            policy = np.argmax(Q, axis=1)
            print_policy_arrows(policy)
    
    # Construir política greedy final
    policy = np.argmax(Q, axis=1)
    print("\nPolítica final (acciones por estado):")
    print_policy_arrows(policy)
    
    # Nuevo entorno para visualizar
    view_env = gym.make("FrozenLake-v1", is_slippery=True, map_name="4x4", render_mode='human')
    env.render()

    while True:
        render_policy(view_env, policy, delay=0.4)
        time.sleep(1)

def print_policy_arrows(policy):    # funciona solo en grid quadrado
    action_map = { 0: "↑", 1: "↓", 2: "←", 3: "→" }
    nS = len(policy)
    side = int(np.sqrt(nS))
    grid = [action_map[a] for a in policy]
    for i in range(side):
        print(" ".join(grid[i*side:(i+1)*side]))
    print("\n")  # Separador

if __name__ == "__main__":
    main()