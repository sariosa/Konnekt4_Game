# play_console.py
# Console play for Connect 4:
# - Human vs Human
# - Human vs Random
# - Human vs Heuristic
# - Human vs Q-learning
#
# Required files in same folder:
#   connect4_env.py
#   policies.py
#   qlearning.py

from connect4_env import EnvConnect4
from policies import PolicyRandom, PolicyHeuristic
from qlearning import train, evaluate, QLearningAgent


def get_qlearning_action(agent: QLearningAgent, env):
    """
    Convert the environment board into the raw-board format expected
    by the old tabular Q-learning agent, then choose an action.
    """
    import numpy as np

    # Convert env.board (row-major, top row first) into the board shape
    # expected by qlearning.py utilities
    board = np.array(env.board, dtype=np.int8).reshape(env.num_rows, env.num_cols)

    # In the old qlearning.py, the agent always acts as turn=1
    legal = env._get_legal_actions()
    return agent.choose_action(board, turn=1, legal=legal)


def play(env, opponents_policy=None, show_q_values: bool = False):
    keep_playing = True

    while keep_playing:
        observation, info = env.reset()
        episode_over = False

        while not episode_over:
            env.print_current_board()

            # Human turn (Player 1 - X)
            if env.turn == 1:
                legal = env._get_legal_actions()
                action = int(
                    input(
                        f"Your (valid) move (column 0-6) as player "
                        f"{env.pos_value_to_name[env.turn]}: "
                    )
                )
                while action not in legal:
                    print(f"Invalid. Legal columns: {legal}")
                    action = int(input("Try again: "))

                observation, reward, terminated, truncated, info = env.step(action)
                episode_over = terminated or truncated
                if episode_over:
                    break

            # Opponent turn (Player 2 - O)
            if opponents_policy is None:
                # Human vs Human
                env.print_current_board()
                legal = env._get_legal_actions()
                action = int(
                    input(
                        f"Other player's (valid) move (column 0-6) as player "
                        f"{env.pos_value_to_name[env.turn]}: "
                    )
                )
                while action not in legal:
                    print(f"Invalid. Legal columns: {legal}")
                    action = int(input("Try again: "))

            else:
                # Policy-controlled opponent
                if isinstance(opponents_policy, QLearningAgent):
                    action = get_qlearning_action(opponents_policy, env)
                else:
                    action = opponents_policy._get_action(env, observation)

                print(
                    f"Other player's ({env.pos_value_to_name[env.turn]}) move: "
                    f"{action} ({env.col_id_to_name[action]})"
                )

                # Show Q-values for old tabular Q-learning agent
                if show_q_values and isinstance(opponents_policy, QLearningAgent):
                    import numpy as np
                    board = np.array(env.board, dtype=np.int8).reshape(env.num_rows, env.num_cols)
                    key = opponents_policy.state_key(board, 1)
                    if key in opponents_policy.q:
                        legal_actions = env._get_legal_actions()
                        action_info = {
                            env.col_id_to_name[a]: f"{float(opponents_policy.q[key][a]):.4f}"
                            for a in legal_actions
                        }
                        print("Q-values:", action_info)

            observation, reward, terminated, truncated, info = env.step(action)
            episode_over = terminated or truncated

        # Game ended
        env.print_current_board()

        if env.is_winner(mark=1):
            result = "X wins!"
        elif env.is_winner(mark=2):
            result = "O wins!"
        else:
            result = "Draw"

        print(f"=> Result: {result}")
        env.close()

        user_input = input("Continue playing (y=yes, n=no): ").strip().lower()
        keep_playing = user_input == "y"


def main_menu():
    print("\n=== Connect 4 Console ===")
    print("1) Human vs Human")
    print("2) Human vs Random")
    print("3) Human vs Heuristic")
    print("4) Train Q-learning (200k) then Human vs Q-learning")
    print("5) Exit")

    choice = input("Choose option (1-5): ").strip()
    return choice


if __name__ == "__main__":
    while True:
        choice = main_menu()

        if choice == "1":
            play(env=EnvConnect4(), opponents_policy=None)

        elif choice == "2":
            play(env=EnvConnect4(), opponents_policy=PolicyRandom())

        elif choice == "3":
            play(env=EnvConnect4(), opponents_policy=PolicyHeuristic())

        elif choice == "4":
            print("\nTraining Q-learning agent for 200000 episodes...")
            agent = train(episodes=200000)
            evaluate(q_file="q_table.pkl", games=1000, out_plot="evaluation.png")
            agent.epsilon = 0.0
            play(env=EnvConnect4(), opponents_policy=agent, show_q_values=False)

        elif choice == "5":
            print("Bye!")
            break

        else:
            print("Invalid choice. Please select 1-5.")