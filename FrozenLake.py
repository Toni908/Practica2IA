""" 
Monte Carlo para Frozen Lake

Autor: Antonio Garcia Font
"""

import time
import gymnasium as gym
import numpy as np
import random

def generate_episode(env, Q, epsilon):
    episode = []
    state, info = env.reset()
    done = False
    
    while not done:
        
        # Política epsilon-greedy
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        episode.append((state, action, reward))
        state = next_state
    
    return episode

def render_policy(env, policy, delay=0.4):
    state, _ = env.reset()
    done = False

    while not done:
        env.render()
        action = policy[state]
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        time.sleep(delay)

    env.render()
    print(f"Recompensa final: {reward}")

def main():
    env = gym.make("FrozenLake-v1", is_slippery=True, map_name="4x4")
    
    gamma = 0.9
    epsilon = 0.1
    episodes = 50000
    
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    returns = {}
    
    for ep in range(episodes):
        
        episode = generate_episode(env, Q, epsilon)
        
        G = 0
        visited = set()
        
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward
            
            if (state, action) not in visited:
                visited.add((state, action))
                
                if (state, action) not in returns:
                    returns[(state, action)] = []
                
                returns[(state, action)].append(G)
                
                Q[state, action] = np.mean(returns[(state, action)])
        
        if ep % 2000 == 0:
            print(f"Episodio {ep} | Política parcial:")
            print(np.argmax(Q, axis=1).reshape(4, 4))

    # YA hemos terminado el entrenamiento: construir política greedy
    policy = np.argmax(Q, axis=1)

    # Mostrar la política final en formato grid
    nS = env.observation_space.n               # <<< FIX
    side = int(np.sqrt(nS))
    print("Política final (acciones por estado):")
    print(policy.reshape(side, side))

    # Nuevo entorno para visualizar
    view_env = gym.make("FrozenLake-v1", is_slippery=True, map_name="4x4", render_mode='human')

    while True:
        render_policy(view_env, policy, delay=0.4)
        time.sleep(1)
 
    while True:
        render_policy(env, policy, delay=0.4)
        time.sleep(1)

main()