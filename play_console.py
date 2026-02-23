# play_console.py
#
# Console play for Connect 4:
# - Human vs Human
# - Human vs Random
# - Human vs Q-learning
# - Human vs SARSA
#
# Make sure these files exist in the same folder:
#   connect4_env.py
#   policies.py
#   train_qlearning.py
#   train_sarsa.py

from connect4_env import EnvConnect4
from policies import PolicyRandom
from train_qlearning import train_q_learning_vs_random
from train_sarsa import train_sarsa_vs_random


def play(env, opponents_policy=None, show_q_values: bool = False):
    keep_playing = True

    while keep_playing:
        observation, info = env.reset()
        episode_over = False

        while not episode_over:
            env.print_current_board()

            # ---------- Human turn (X) ----------
            if env.turn == 1:
                legal = env._get_legal_actions()
                action = int(input(f"Your (valid) move (column 0-6) as player {env.pos_value_to_name[env.turn]}: "))
                while action not in legal:
                    print(f"Invalid. Legal columns: {legal}")
                    action = int(input("Try again: "))

                observation, reward, terminated, truncated, info = env.step(action)
                episode_over = terminated or truncated
                if episode_over:
                    break

            # ---------- Opponent turn (O) ----------
            if opponents_policy is None:
                env.print_current_board()
                legal = env._get_legal_actions()
                action = int(input(f"Other player's (valid) move (column 0-6) as player {env.pos_value_to_name[env.turn]}: "))
                while action not in legal:
                    print(f"Invalid. Legal columns: {legal}")
                    action = int(input("Try again: "))

            else:
                action = opponents_policy._get_action(env, observation)
                print(f"Other player's ({env.pos_value_to_name[env.turn]}) move: {action} ({env.col_id_to_name[action]})")

                # Optional: show Q-values (only for Q-table policies)
                if show_q_values and hasattr(opponents_policy, "Q"):
                    s = tuple(observation["board"]) + (int(observation["turn"]),)
                    if s in opponents_policy.Q:
                        legal_actions = env._get_legal_actions()
                        action_info = {
                            env.col_id_to_name[a]: f"{float(opponents_policy.Q[s][a]):.4f}"
                            for a in legal_actions
                        }
                        print(action_info)

            observation, reward, terminated, truncated, info = env.step(action)
            episode_over = terminated or truncated

        # ---------- Game ended ----------
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
        keep_playing = True if user_input == "y" else False


def main_menu():
    print("\n=== Connect 4 Console ===")
    print("1) Human vs Human")
    print("2) Human vs Random")
    print("3) Train Q-learning then Human vs Q-learning")
    print("4) Train SARSA then Human vs SARSA")
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
            env = EnvConnect4()
            agent, returns = train_q_learning_vs_random(env, episodes=50000)
            play(env=EnvConnect4(), opponents_policy=agent, show_q_values=False)

        elif choice == "4":
            env = EnvConnect4()
            agent, returns = train_sarsa_vs_random(env, episodes=50000)
            play(env=EnvConnect4(), opponents_policy=agent, show_q_values=False)

        elif choice == "5":
            print("Bye!")
            break

        else:
            print("Invalid choice. Please select 1-5.")