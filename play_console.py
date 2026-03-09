from connect4_env import EnvConnect4
from policies import PolicyRandom, PolicyHeuristic
from qlearning import train


def play(env, opponents_policy=None, show_q_values: bool = False):
    """
    Runs a console-based Connect 4 game.

    The function supports the following modes:
    - Human vs Human
    - Human vs policy-controlled opponent

    Parameters
    ----------
    env : EnvConnect4
        Connect 4 environment.
    opponents_policy : object, optional
        Opponent policy. If None, the game is played in Human vs Human mode.
    show_q_values : bool, optional
        If True, Q-values of the Q-learning policy are displayed for legal actions.
    """
    keep_playing = True

    while keep_playing:
        observation, info = env.reset()
        episode_over = False

        while not episode_over:
            env.print_current_board()

            # Player 1 is always controlled by the human
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

            # Player 2 can be either another human or a policy-controlled opponent
            if opponents_policy is None:
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
                action = opponents_policy._get_action(env, observation)
                print(
                    f"Other player's ({env.pos_value_to_name[env.turn]}) move: "
                    f"{action} ({env.col_id_to_name[action]})"
                )

                # Optionally display Q-values for the current state
                if show_q_values and hasattr(opponents_policy, "Q"):
                    s = tuple(observation["board"]) + (int(observation["turn"]),)
                    if s in opponents_policy.Q:
                        legal_actions = env._get_legal_actions()
                        action_info = {
                            env.col_id_to_name[a]: f"{float(opponents_policy.Q[s][a]):.4f}"
                            for a in legal_actions
                        }
                        print("Q-values:", action_info)

            observation, reward, terminated, truncated, info = env.step(action)
            episode_over = terminated or truncated

        # Final board and result after the game ends
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
    """
    Displays the console menu and returns the user's choice.

    Returns
    -------
    str
        Selected menu option.
    """
    print("\n=== Connect 4 Console ===")
    print("1) Human vs Human")
    print("2) Human vs Random")
    print("3) Human vs Heuristic")
    print("4) Train Q-learning then Human vs Q-learning")
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
            env = EnvConnect4()
            agent = train(episodes=200000)
            play(env=EnvConnect4(), opponents_policy=agent, show_q_values=False)

        elif choice == "5":
            print("Bye!")
            break

        else:
            print("Invalid choice. Please select 1-5.")
