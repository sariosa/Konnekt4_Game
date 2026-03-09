import random
from qlearning import PolicyQLearningV4
from connect4_env import EnvConnect4
from policies import PolicyHeuristic, PolicyRandom


def load_agent(q_file="q_table.pkl", seed=42):
    """
    Loads a trained Q-learning agent from file.

    Parameters
    ----------
    q_file : str, optional
        Filename of the saved Q-table.
    seed : int, optional
        Random seed used to initialize the agent.

    Returns
    -------
    tuple
        Loaded agent and environment.
    """
    env = EnvConnect4()
    agent = PolicyQLearningV4(env, seed=seed)
    agent.load(q_file)
    agent.epsilon = 0.0
    return agent, env


def evaluate_vs_policy(agent, env, opponent_policy, games=500, mode="alternate"):
    """
    Evaluates the agent against a fixed opponent policy.

    Parameters
    ----------
    agent : PolicyQLearningV4
        Trained Q-learning agent.
    env : EnvConnect4
        Connect 4 environment.
    opponent_policy : object
        Opponent policy used for evaluation.
    games : int, optional
        Number of evaluation games.
    mode : str, optional
        Starting mode:
        - "alternate": players alternate who starts
        - "first": agent always starts first
        - "second": agent always starts second

    Returns
    -------
    dict
        Dictionary containing wins, losses, draws, and rates.
    """
    W = L = D = 0

    for i in range(games):
        obs, info = env.reset()
        done = False

        if mode == "alternate":
            agent_is_p1 = (i % 2 == 0)
        elif mode == "first":
            agent_is_p1 = True
        elif mode == "second":
            agent_is_p1 = False
        else:
            raise ValueError("mode must be 'alternate', 'first', or 'second'")

        while not done:
            is_agent_turn = (env.turn == 1 and agent_is_p1) or (env.turn == 2 and not agent_is_p1)

            if is_agent_turn:
                action = agent._get_action(env, obs)
            else:
                action = opponent_policy._get_action(env, obs)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        agent_piece = 1 if agent_is_p1 else 2

        if info["winner"] == agent_piece:
            W += 1
        elif info["is_draw"]:
            D += 1
        else:
            L += 1

    return {
        "W": W,
        "L": L,
        "D": D,
        "win_rate": W / games,
        "draw_rate": D / games,
        "loss_rate": L / games,
    }


def evaluate_vs_training_mix(agent, env, games=500, heuristic_prob=0.80, seed=42, mode="alternate"):
    """
    Evaluates the agent against a mixed opponent setting.

    In this setting, the opponent plays heuristically with a given
    probability and randomly otherwise.

    Parameters
    ----------
    agent : PolicyQLearningV4
        Trained Q-learning agent.
    env : EnvConnect4
        Connect 4 environment.
    games : int, optional
        Number of evaluation games.
    heuristic_prob : float, optional
        Probability of selecting the heuristic opponent policy.
    seed : int, optional
        Random seed for opponent selection.
    mode : str, optional
        Starting mode:
        - "alternate": players alternate who starts
        - "first": agent always starts first
        - "second": agent always starts second

    Returns
    -------
    dict
        Dictionary containing wins, losses, draws, and rates.
    """
    rng = random.Random(seed)
    heuristic = PolicyHeuristic(seed=seed)
    rand_pol = PolicyRandom()

    W = L = D = 0

    for i in range(games):
        obs, info = env.reset()
        done = False

        if mode == "alternate":
            agent_is_p1 = (i % 2 == 0)
        elif mode == "first":
            agent_is_p1 = True
        elif mode == "second":
            agent_is_p1 = False
        else:
            raise ValueError("mode must be 'alternate', 'first', or 'second'")

        while not done:
            is_agent_turn = (env.turn == 1 and agent_is_p1) or (env.turn == 2 and not agent_is_p1)

            if is_agent_turn:
                action = agent._get_action(env, obs)
            else:
                opp = heuristic if rng.random() < heuristic_prob else rand_pol
                action = opp._get_action(env, obs)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        agent_piece = 1 if agent_is_p1 else 2

        if info["winner"] == agent_piece:
            W += 1
        elif info["is_draw"]:
            D += 1
        else:
            L += 1

    return {
        "W": W,
        "L": L,
        "D": D,
        "win_rate": W / games,
        "draw_rate": D / games,
        "loss_rate": L / games,
    }


def print_result(title, result):
    """
    Prints evaluation results in a readable format.

    Parameters
    ----------
    title : str
        Title describing the evaluation setting.
    result : dict
        Dictionary containing wins, losses, draws, and rates.
    """
    print(f"\n{title}")
    print(f"  W/L/D = {result['W']}/{result['L']}/{result['D']}")
    print(f"  Win rate  = {result['win_rate']:.3f}")
    print(f"  Draw rate = {result['draw_rate']:.3f}")
    print(f"  Loss rate = {result['loss_rate']:.3f}")


def main():
    """
    Loads the trained agent and runs several evaluation settings.
    """
    q_file = "q_table.pkl"
    games = 500
    seed = 42

    agent, env = load_agent(q_file=q_file, seed=seed)

    # Evaluation against a random opponent
    res_random = evaluate_vs_policy(
        agent, env, PolicyRandom(), games=games, mode="alternate"
    )
    print_result("Against Random (alternate starts)", res_random)

    # Evaluation against a heuristic opponent
    res_heur = evaluate_vs_policy(
        agent, env, PolicyHeuristic(seed=999), games=games, mode="alternate"
    )
    print_result("Against Heuristic (alternate starts)", res_heur)

    # Evaluation against the mixed opponent setting used during training
    res_mix = evaluate_vs_training_mix(
        agent, env, games=games, heuristic_prob=0.80, seed=seed, mode="alternate"
    )
    print_result("Against Training Mix (alternate starts)", res_mix)

    # Evaluation against heuristic when the agent always plays second
    res_second = evaluate_vs_policy(
        agent, env, PolicyHeuristic(seed=999), games=games, mode="second"
    )
    print_result("Against Heuristic (agent always second)", res_second)

    # Evaluation against heuristic when the agent always plays first
    res_first = evaluate_vs_policy(
        agent, env, PolicyHeuristic(seed=999), games=games, mode="first"
    )
    print_result("Against Heuristic (agent always first)", res_first)


if __name__ == "__main__":
    main()
