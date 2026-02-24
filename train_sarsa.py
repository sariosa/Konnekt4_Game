import numpy as np
from connect4_env import EnvConnect4
from policies import PolicyQLearning, PolicyRandom


def train_sarsa_vs_random(
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

    agent = PolicyQLearning(env, alpha=alpha, gamma=gamma, epsilon=epsilon_start, seed=seed)
    opponent = PolicyRandom()

    returns = []
    eps = epsilon_start

    for ep in range(episodes):
        observation, info = env.reset(seed=seed + ep)
        done = False
        G = 0.0

        while not done:

            if env.turn != 1:
                a_opp = opponent._get_action(env, observation)
                observation, _, terminated, truncated, info = env.step(a_opp)
                done = terminated or truncated
                continue

            s = tuple(observation["board"]) + (int(observation["turn"]),)
            a = agent._get_action(env, observation)

            obs_next, r, terminated, truncated, info_next = env.step(a)
            done = terminated or truncated

            s_next = tuple(obs_next["board"]) + (int(obs_next["turn"]),)

            if s_next not in agent.Q:
                agent.Q[s_next] = np.zeros(agent.n_actions, dtype=np.float32)

            if done:
                target = r
            else:
                a_next = agent._get_action(env, obs_next)
                target = r + gamma * float(agent.Q[s_next][a_next])

            agent.Q[s][a] += alpha * (target - agent.Q[s][a])

            observation = obs_next
            G += r

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
    agent, returns = train_sarsa_vs_random(env)
    print("Training finished. Q-table states:", len(agent.Q))
