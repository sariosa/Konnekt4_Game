import numpy as np
from connect4_env import EnvConnect4
from policies import PolicyQLearning, PolicyRandom


def train_q_learning_vs_random(
    env: EnvConnect4,
    episodes: int = 50000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.9995,
    seed: int = 7,
    log_every: int = 5000,
):
    """
    Trains a Q-learning agent as X (turn=1) against a random opponent O (turn=2).

    Reward used for learning:
    - env step reward: -0.01
    - terminal shaping (added on top):
        +1.0 if X wins
        -1.0 if O wins
        +0.5 if draw
    """
    agent = PolicyQLearning(env, alpha=alpha, gamma=gamma, epsilon=epsilon_start, seed=seed)
    opponent = PolicyRandom()

    returns = []
    eps = epsilon_start

    for ep in range(episodes):
        observation, info = env.reset(seed=seed + ep)
        done = False
        G = 0.0

        while not done:
            # X (agent) move
            if env.turn != 1:
                a_opp = opponent._get_action(env, observation)
                observation, _, terminated, truncated, info = env.step(a_opp)
                done = terminated or truncated
                continue

            s = tuple(observation["board"]) + (int(observation["turn"]),)

            a = agent._get_action(env, observation)
            obs_after_x, r_step_x, terminated, truncated, info_after_x = env.step(a)
            done = terminated or truncated

            # If game ends after X move
            if done:
                if env.is_winner(mark=1):
                    r_terminal = 1.0
                elif env.is_winner(mark=2):
                    r_terminal = -1.0
                else:
                    r_terminal = 0.5

                r_total = float(r_step_x) + float(r_terminal)
                s_next = tuple(obs_after_x["board"]) + (int(obs_after_x["turn"]),)
                agent.update(s, a, r_total, s_next, info_after_x["legal columns"], done=True)
                G += r_total
                break

            # O (random opponent) move
            a_opp = opponent._get_action(env, obs_after_x)
            obs_after_o, _, terminated2, truncated2, info_after_o = env.step(a_opp)
            done = terminated2 or truncated2

            # Reward from X perspective after opponent response
            if done:
                if env.is_winner(mark=2):
                    r_after = -1.0
                else:
                    r_after = 0.5
            else:
                r_after = 0.0

            r_total = float(r_step_x) + float(r_after)

            s_next = tuple(obs_after_o["board"]) + (int(obs_after_o["turn"]),)
            agent.update(s, a, r_total, s_next, info_after_o["legal columns"], done=done)

            observation, info = obs_after_o, info_after_o
            G += r_total

        eps = max(epsilon_end, eps * epsilon_decay)
        agent.epsilon = eps
        returns.append(G)

        if log_every and (ep + 1) % log_every == 0:
            print(
                f"Episode {ep+1}/{episodes} | epsilon={agent.epsilon:.3f} "
                f"| avg return(last {log_every})={np.mean(returns[-log_every:]):.3f}"
            )

    return agent, returns


if __name__ == "__main__":
    env = EnvConnect4()
    agent, returns = train_q_learning_vs_random(env, episodes=50000)
    print("Training finished. Q-table states:", len(agent.Q))