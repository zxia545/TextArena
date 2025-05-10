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
        # "Poker-v0",
        # "Checkers-v0",
        # "Othello-v0",
        # "Nim-v0",
        # "MemoryGame-v0",
        # "Snake-v0",
        # "WordChains-v0"
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
                player_names={0: "Player0", 1: "Player1"},
            )

            env.reset(num_players=len(agents))
            done = False
            while not done:
                player_id, observation = env.get_observation()
                action = agents[player_id](observation)
                done, info = env.step(action=action)
            
            rewards = env.close()
            results[game].append(rewards)
            
            # Determine game outcome
            if rewards[0] > rewards[1]:
                outcome = "Player 0 won"
            elif rewards[1] > rewards[0]:
                outcome = "Player 1 won"
            else:
                outcome = "Draw"
            
            # Log results with outcome
            with open("setting1_results.jsonl", "a") as f:
                json.dump({
                    "game": game,
                    "game_num": game_num,
                    "rewards": rewards,
                    "outcome": outcome
                }, f)
                f.write("\n")

# Setting 2: Player1 with history
def setting2_player1_with_history():
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
        # "Poker-v0",
        # "Checkers-v0",
        # "Othello-v0",
        # "Nim-v0",
        # "MemoryGame-v0",
        # "Snake-v0",
        # "WordChains-v0"
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
                player_names={0: "Player0", 1: "Player1"},
            )

            # Add previous game history to Player1's prompt if available
            if game_num > 0:
                history_prompt = "Previous game history:\n"
                for prev_game_num, prev_game in enumerate(game_histories[game]):
                    history_prompt += f"\nGame {prev_game_num + 1}:\n"
                    history_prompt += f"Outcome: {prev_game['outcome']}\n"
                    history_prompt += "Move sequence:\n"
                    for move in prev_game["moves"]:
                        history_prompt += f"- Player {move['player']}: {move['action']} -> {move['result']}\n"
                
                agents[1].system_prompt = history_prompt + "\n" + agents[1].system_prompt

            env.reset(num_players=len(agents))
            current_game_history = {
                "moves": [],
                "outcome": None
            }
            
            done = False
            while not done:
                player_id, observation = env.get_observation()
                action = agents[player_id](observation)
                done, info = env.step(action=action)
                
                # Record move with more context
                current_game_history["moves"].append({
                    "player": player_id,
                    "observation": observation,
                    "action": action,
                    "result": info.get("result", "No result recorded")
                })
            
            rewards = env.close()
            
            # Record game outcome
            if rewards[0] > rewards[1]:
                current_game_history["outcome"] = "Player 0 won"
            elif rewards[1] > rewards[0]:
                current_game_history["outcome"] = "Player 1 won"
            else:
                current_game_history["outcome"] = "Draw"
            
            game_histories[game].append(current_game_history)
            
            # Log results
            with open("setting2_results.jsonl", "a") as f:
                json.dump({
                    "game": game,
                    "game_num": game_num,
                    "rewards": rewards,
                    "history": current_game_history
                }, f)
                f.write("\n")

# Setting 3: Player0 as master teacher for Player1
def setting3_player0_teacher():
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
        # "Poker-v0",
        # "Checkers-v0",
        # "Othello-v0",
        # "Nim-v0",
        # "MemoryGame-v0",
        # "Snake-v0",
        # "WordChains-v0"
    ]

    # Store game histories and learnings
    game_histories = {game: [] for game in selected_games}
    game_learnings = {game: [] for game in selected_games}
    
    # Play each game 5 times
    for game in selected_games:
        print(f"Playing {game}...")
        for game_num in range(5):
            env = ta.make(env_id=game)
            env = ta.wrappers.LLMObservationWrapper(env=env)
            env = ta.wrappers.SimpleRenderWrapper(
                env=env,
                player_names={0: "Player0", 1: "Player1"},
            )

            # If not first game, add previous learnings to Player1's prompt
            if game_num > 0:
                learning_prompt = "Previous game learnings:\n"
                for prev_game_num, learning in enumerate(game_learnings[game]):
                    learning_prompt += f"\nGame {prev_game_num + 1} learnings:\n{learning}\n"
                
                agents[1].system_prompt = learning_prompt + "\n" + agents[1].system_prompt

            env.reset(num_players=len(agents))
            current_game_history = {
                "moves": [],
                "outcome": None
            }
            
            done = False
            while not done:
                player_id, observation = env.get_observation()
                action = agents[player_id](observation)
                done, info = env.step(action=action)
                
                # Record move with more context
                current_game_history["moves"].append({
                    "player": player_id,
                    "observation": observation,
                    "action": action,
                    "result": info.get("result", "No result recorded")
                })
            
            rewards = env.close()
            
            # Record game outcome
            if rewards[0] > rewards[1]:
                current_game_history["outcome"] = "Player 0 won"
            elif rewards[1] > rewards[0]:
                current_game_history["outcome"] = "Player 1 won"
            else:
                current_game_history["outcome"] = "Draw"
            
            # Generate key learnings from the game
            learning_prompt = (
                f"As a master teacher, analyze this game and provide key learnings for Player1.\n"
                f"Game outcome: {current_game_history['outcome']}\n"
                f"Move sequence: {json.dumps(current_game_history['moves'], indent=2)}\n\n"
                f"Provide learnings in this format:\n"
                f"1. Strategic principles to follow\n"
                f"2. Specific moves to consider\n"
                f"3. Moves to avoid\n"
                f"4. Key patterns to watch for"
            )
            current_learning = agents[0](learning_prompt)
            game_learnings[game].append(current_learning)
            
            game_histories[game].append(current_game_history)
            
            # Log results
            with open("setting3_results.jsonl", "a") as f:
                json.dump({
                    "game": game,
                    "game_num": game_num,
                    "rewards": rewards,
                    "history": current_game_history,
                    "learning": current_learning
                }, f)
                f.write("\n")

if __name__ == "__main__":
    # Run the experiment you want
    setting1_basic_evaluation()
    setting2_player1_with_history()
    setting3_player0_teacher() 