import textarena as ta
import json
from typing import Dict, List
import os

# Setting 1: Basic win rate evaluation with TrueSkill
def setting1_basic_evaluation():
    # Initialize agents with custom API base and SSH key
    agents = {
        0: ta.agents.OpenRouterAgent(
            model_name="llama-3.1-8b",
            api_base="http://localhost:8010/v1",  # Custom API base
            api_key="your_api_key_here"  # SSH key
        ),
        1: ta.agents.OpenRouterAgent(
            model_name="llama-3.1-8b",
            api_base="http://localhost:8010/v1",
            api_key="your_api_key_here"
        ),
    }

    # Select 8 games for evaluation
    selected_games = [
        "TicTacToe-v0",
        "Poker-v0",
        "Checkers-v0",
        "Othello-v0",
        "Nim-v0",
        "MemoryGame-v0",
        "Snake-v0",
        "WordChains-v0"
    ]

    # Track results for TrueSkill calculation
    results = {game: [] for game in selected_games}
    
    # Play each game 5 times
    for game in selected_games:
        print(f"Playing {game}...")
        for game_num in range(5):
            env = ta.make(env_id=game)
            env = ta.wrappers.LLMObservationWrapper(env=env)
            env = ta.wrappers.SimpleRenderWrapper(
                env=env,
                player_names={0: "Player1", 1: "Player2"},
            )

            env.reset(num_players=len(agents))
            done = False
            while not done:
                player_id, observation = env.get_observation()
                action = agents[player_id](observation)
                done, info = env.step(action=action)
            
            rewards = env.close()
            results[game].append(rewards)
            
            # Log results
            with open("setting1_results.json", "a") as f:
                json.dump({
                    "game": game,
                    "game_num": game_num,
                    "rewards": rewards
                }, f)
                f.write("\n")

# Setting 2: Player2 with history
def setting2_player2_with_history():
    agents = {
        0: ta.agents.OpenRouterAgent(
            model_name="llama-3.1-8b",
            api_base="http://localhost:8010/v1",
            api_key="your_api_key_here"
        ),
        1: ta.agents.OpenRouterAgent(
            model_name="llama-3.1-8b",
            api_base="http://localhost:8010/v1",
            api_key="your_api_key_here"
        ),
    }

    selected_games = [
        "TicTacToe-v0",
        "Poker-v0",
        "Checkers-v0",
        "Othello-v0",
        "Nim-v0",
        "MemoryGame-v0",
        "Snake-v0",
        "WordChains-v0"
    ]

    # Store game histories
    game_histories = {game: [] for game in selected_games}
    
    # Play each game 5 times
    for game in selected_games:
        print(f"Playing {game}...")
        for game_num in range(5):
            env = ta.make(env_id=game)
            env = ta.wrappers.LLMObservationWrapper(env=env)
            env = ta.wrappers.SimpleRenderWrapper(
                env=env,
                player_names={0: "Player1", 1: "Player2"},
            )

            # Add previous game history to Player2's prompt if available
            if game_num > 0:
                history_prompt = f"Previous game history:\n{json.dumps(game_histories[game][-1], indent=2)}\n\n"
                agents[1].system_prompt = history_prompt + agents[1].system_prompt

            env.reset(num_players=len(agents))
            current_game_history = []
            
            done = False
            while not done:
                player_id, observation = env.get_observation()
                action = agents[player_id](observation)
                done, info = env.step(action=action)
                
                # Record game history
                current_game_history.append({
                    "player": player_id,
                    "observation": observation,
                    "action": action
                })
            
            rewards = env.close()
            game_histories[game].append(current_game_history)
            
            # Log results
            with open("setting2_results.json", "a") as f:
                json.dump({
                    "game": game,
                    "game_num": game_num,
                    "rewards": rewards,
                    "history": current_game_history
                }, f)
                f.write("\n")

# Setting 3: Player1 gives advice to Player2
def setting3_player1_advice():
    agents = {
        0: ta.agents.OpenRouterAgent(
            model_name="llama-3.1-8b",
            api_base="http://localhost:8010/v1",
            api_key="your_api_key_here"
        ),
        1: ta.agents.OpenRouterAgent(
            model_name="llama-3.1-8b",
            api_base="http://localhost:8010/v1",
            api_key="your_api_key_here"
        ),
    }

    selected_games = [
        "TicTacToe-v0",
        "Poker-v0",
        "Checkers-v0",
        "Othello-v0",
        "Nim-v0",
        "MemoryGame-v0",
        "Snake-v0",
        "WordChains-v0"
    ]

    # Store game histories and advice
    game_histories = {game: [] for game in selected_games}
    player1_advice = {game: [] for game in selected_games}
    
    # Play each game 5 times
    for game in selected_games:
        print(f"Playing {game}...")
        for game_num in range(5):
            env = ta.make(env_id=game)
            env = ta.wrappers.LLMObservationWrapper(env=env)
            env = ta.wrappers.SimpleRenderWrapper(
                env=env,
                player_names={0: "Player1", 1: "Player2"},
            )

            # If not first game, Player1 gives advice based on previous game
            if game_num > 0:
                advice_prompt = f"Based on the previous game history, what advice would you give to Player2 for the next game?\nPrevious game:\n{json.dumps(game_histories[game][-1], indent=2)}"
                advice = agents[0](advice_prompt)
                player1_advice[game].append(advice)
                
                # Add advice to Player2's prompt
                agents[1].system_prompt = f"Advice from Player1: {advice}\n\n" + agents[1].system_prompt

            env.reset(num_players=len(agents))
            current_game_history = []
            
            done = False
            while not done:
                player_id, observation = env.get_observation()
                action = agents[player_id](observation)
                done, info = env.step(action=action)
                
                # Record game history
                current_game_history.append({
                    "player": player_id,
                    "observation": observation,
                    "action": action
                })
            
            rewards = env.close()
            game_histories[game].append(current_game_history)
            
            # Log results
            with open("setting3_results.json", "a") as f:
                json.dump({
                    "game": game,
                    "game_num": game_num,
                    "rewards": rewards,
                    "history": current_game_history,
                    "advice": player1_advice[game][-1] if game_num > 0 else None
                }, f)
                f.write("\n")

if __name__ == "__main__":
    # Run the experiment you want
    setting1_basic_evaluation()
    # setting2_player2_with_history()
    # setting3_player1_advice() 