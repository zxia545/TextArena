import textarena as ta
import json
from typing import Dict, List
import os
import argparse
import subprocess
import sys
import time
from this_utils import start_vllm_server, stop_vllm_server
from textarena.agents.basic_agents import Qwen3Agent


import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename='experiment.log')
logger = logging.getLogger(__name__)

def get_player1_agent(model_name: str):
    """Helper function to create Player 1 agent with specified model"""
    return ta.agents.OpenRouterAgent(
        model_name=model_name,
        api_base="http://localhost:8010/v1",
        api_key="your_api_key_here",
        timeout=60
    )

def get_player0_agent():
    """Helper function to create Player 0 agent (Qwen2.5-32B)"""
    return ta.agents.OpenRouterAgent(
        model_name="qwen2.5-32b-chat",
        api_base="http://localhost:8020/v1",
        api_key="your_api_key_here",
        timeout=60
    )

# # List of all games to evaluate
# ALL_GAMES = [
#     "SpellingBee-v0",
#     "Poker-v0",
#     "SpiteAndMalice-v0",
#     "Stratego-v0",
#     "Tak-v0",
#     "TruthAndDeception-v0",
#     "UltimateTicTacToe-v0",
#     "WordChains-v0",
#     "TicTacToe-v0",
#     "Breakthrough-v0",
#     "Checkers-v0",
#     "KuhnPoker-v0",
#     "LetterAuction-v0",
#     "MemoryGame-v0",
#     "Nim-v0",
#     "Othello-v0",
#     "PigDice-v0",
#     "SimpleBlindAuction-v0",
#     "Snake-v0",
#     "SecretMafia-v0",
# ]

# Setting 1: Basic win rate evaluation with TrueSkill
def setting1_basic_evaluation(player1_model_name: str, selected_games: List[str]):
    logger.info(f"Starting Setting 1 evaluation with Player1 model: {player1_model_name}")
    # Initialize agents
    agents = {
        0: get_player0_agent(),
        1: get_player1_agent(player1_model_name)
    }
    logger.info("Agents initialized successfully")

    # Track results for TrueSkill calculation
    results = {game: [] for game in selected_games}
    
    # Play each game 10 times
    for game in selected_games:
        logger.info(f"Starting {game} evaluation with {player1_model_name}")
        for game_num in range(10):
            try:
                logger.info(f"Starting game {game_num + 1}/10 for {game}")
                env = ta.make(env_id=game)
                env = ta.wrappers.LLMObservationWrapper(env=env)
                env = ta.wrappers.SimpleRenderWrapper(
                    env=env,
                    player_names={0: "Player0", 1: "Player1"},
                )
                logger.debug(f"Environment created and configured for {game}")

                env.reset(num_players=len(agents))
                done = False
                move_count = 0
                while not done:
                    try:
                        player_id, observation = env.get_observation()
                        logger.debug(f"Player {player_id}'s turn - Move {move_count + 1}")
                        action = agents[player_id](observation)
                        logger.debug(f"Player {player_id} action: {action}")
                        done, info = env.step(action=action)
                        move_count += 1
                    except Exception as e:
                        logger.error(f"Error during game play: {str(e)}")
                        # If there's an error during gameplay, mark the game as done and move to next
                        done = True
                        rewards = [0, 0]  # Default rewards for failed game
                        break
                
                rewards = env.close()
                results[game].append(rewards)
                logger.info(f"Game {game_num + 1} completed. Rewards: Player0={rewards[0]}, Player1={rewards[1]}")
                
                # Determine game outcome
                if rewards[0] > rewards[1]:
                    outcome = "Player 0 won"
                elif rewards[1] > rewards[0]:
                    outcome = "Player 1 won"
                else:
                    outcome = "Draw"
                logger.info(f"Game {game_num + 1} outcome: {outcome}")
                
                # Log results with outcome and model names
                result_file = f"results/setting1_{player1_model_name}_results.jsonl"
                with open(result_file, "a") as f:
                    json.dump({
                        "game": game,
                        "game_num": game_num,
                        "rewards": rewards,
                        "outcome": outcome,
                        "player0_model": "qwen2.5-32b-chat",
                        "player1_model": player1_model_name,
                        "error": None
                    }, f)
                    f.write("\n")
                logger.debug(f"Results saved to {result_file}")
            except Exception as e:
                logger.error(f"Error in game {game_num + 1} of {game}: {str(e)}")
                # Log the error and continue with next game
                result_file = f"results/setting1_{player1_model_name}_results.jsonl"
                with open(result_file, "a") as f:
                    json.dump({
                        "game": game,
                        "game_num": game_num,
                        "rewards": [0, 0],
                        "outcome": "Error",
                        "player0_model": "qwen2.5-32b-chat",
                        "player1_model": player1_model_name,
                        "error": str(e)
                    }, f)
                    f.write("\n")
                continue

# Setting 2: Player1 with history
def setting2_player1_with_history(player1_model_name: str, selected_games: List[str]):
    logger.info(f"Starting Setting 2 evaluation with Player1 model: {player1_model_name}")
    agents = {
        0: get_player0_agent(),
        1: get_player1_agent(player1_model_name)
    }
    logger.info("Agents initialized successfully")

    # Store game histories
    game_histories = {game: [] for game in selected_games}
    
    # Play each game 10 times
    for game in selected_games:
        logger.info(f"Starting {game} evaluation with {player1_model_name}")
        for game_num in range(10):
            try:
                logger.info(f"Starting game {game_num + 1}/10 for {game}")
                env = ta.make(env_id=game)
                env = ta.wrappers.LLMObservationWrapper(env=env)
                env = ta.wrappers.SimpleRenderWrapper(
                    env=env,
                    player_names={0: "Player0", 1: "Player1"},
                )
                logger.debug(f"Environment created and configured for {game}")

                # Add previous game history to Player1's prompt if available
                if game_num > 0:
                    logger.info(f"Adding history from previous {game_num} games to Player1's prompt")
                    history_prompt = "Previous game history:\n"
                    for prev_game_num, prev_game in enumerate(game_histories[game]):
                        history_prompt += f"\nGame {prev_game_num + 1}:\n"
                        history_prompt += f"Outcome: {prev_game['outcome']}\n"
                        history_prompt += "Move sequence:\n"
                        for move in prev_game["moves"]:
                            history_prompt += f"- Player {move['player']}: {move['action']} -> {move['result']}\n"
                    
                    agents[1].system_prompt = history_prompt + "\n" + agents[1].system_prompt
                    logger.debug("History prompt added to Player1's system prompt")

                env.reset(num_players=len(agents))
                current_game_history = {
                    "moves": [],
                    "outcome": None
                }
                
                done = False
                move_count = 0
                while not done:
                    try:
                        player_id, observation = env.get_observation()
                        logger.debug(f"Player {player_id}'s turn - Move {move_count + 1}")
                        action = agents[player_id](observation)
                        logger.debug(f"Player {player_id} action: {action}")
                        done, info = env.step(action=action)
                        
                        # Record move with more context
                        current_game_history["moves"].append({
                            "player": player_id,
                            "observation": observation,
                            "action": action,
                            "result": info.get("result", "No result recorded")
                        })
                        move_count += 1
                    except Exception as e:
                        logger.error(f"Error during game play: {str(e)}")
                        # If there's an error during gameplay, mark the game as done and move to next
                        done = True
                        rewards = [0, 0]  # Default rewards for failed game
                        current_game_history["outcome"] = "Error"
                        break
                
                rewards = env.close()
                logger.info(f"Game {game_num + 1} completed. Rewards: Player0={rewards[0]}, Player1={rewards[1]}")
                
                # Record game outcome
                if current_game_history["outcome"] != "Error":
                    if rewards[0] > rewards[1]:
                        current_game_history["outcome"] = "Player 0 won"
                    elif rewards[1] > rewards[0]:
                        current_game_history["outcome"] = "Player 1 won"
                    else:
                        current_game_history["outcome"] = "Draw"
                logger.info(f"Game {game_num + 1} outcome: {current_game_history['outcome']}")
                
                game_histories[game].append(current_game_history)
                
                # Log results with model names
                result_file = f"results/setting2_{player1_model_name}_results.jsonl"
                with open(result_file, "a") as f:
                    json.dump({
                        "game": game,
                        "game_num": game_num,
                        "rewards": rewards,
                        "history": current_game_history,
                        "player0_model": "qwen2.5-32b-chat",
                        "player1_model": player1_model_name,
                        "error": None if current_game_history["outcome"] != "Error" else str(e)
                    }, f)
                    f.write("\n")
                logger.debug(f"Results saved to {result_file}")
            except Exception as e:
                logger.error(f"Error in game {game_num + 1} of {game}: {str(e)}")
                # Log the error and continue with next game
                result_file = f"results/setting2_{player1_model_name}_results.jsonl"
                with open(result_file, "a") as f:
                    json.dump({
                        "game": game,
                        "game_num": game_num,
                        "rewards": [0, 0],
                        "history": {"moves": [], "outcome": "Error"},
                        "player0_model": "qwen2.5-32b-chat",
                        "player1_model": player1_model_name,
                        "error": str(e)
                    }, f)
                    f.write("\n")
                continue

# Setting 3: Player0 as master teacher for Player1
def setting3_player0_teacher(player1_model_name: str, selected_games: List[str]):
    logger.info(f"Starting Setting 3 evaluation with Player1 model: {player1_model_name}")
    agents = {
        0: get_player0_agent(),
        1: get_player1_agent(player1_model_name)
    }
    logger.info("Agents initialized successfully")

    # Store game histories and learnings
    game_histories = {game: [] for game in selected_games}
    game_learnings = {game: [] for game in selected_games}
    
    # Play each game 10 times
    for game in selected_games:
        logger.info(f"Starting {game} evaluation with {player1_model_name}")
        for game_num in range(10):
            try:
                logger.info(f"Starting game {game_num + 1}/10 for {game}")
                env = ta.make(env_id=game)
                env = ta.wrappers.LLMObservationWrapper(env=env)
                env = ta.wrappers.SimpleRenderWrapper(
                    env=env,
                    player_names={0: "Player0", 1: "Player1"},
                )
                logger.debug(f"Environment created and configured for {game}")

                # If not first game, add previous learnings to Player1's prompt
                if game_num > 0:
                    logger.info(f"Adding learnings from previous {game_num} games to Player1's prompt")
                    learning_prompt = "Previous game learnings:\n"
                    for prev_game_num, learning in enumerate(game_learnings[game]):
                        learning_prompt += f"\nGame {prev_game_num + 1} learnings:\n{learning}\n"
                    
                    agents[1].system_prompt = learning_prompt + "\n" + agents[1].system_prompt
                    logger.debug("Learning prompt added to Player1's system prompt")

                env.reset(num_players=len(agents))
                current_game_history = {
                    "moves": [],
                    "outcome": None
                }
                
                done = False
                move_count = 0
                while not done:
                    try:
                        player_id, observation = env.get_observation()
                        logger.debug(f"Player {player_id}'s turn - Move {move_count + 1}")
                        action = agents[player_id](observation)
                        logger.debug(f"Player {player_id} action: {action}")
                        done, info = env.step(action=action)
                        
                        # Record move with more context
                        current_game_history["moves"].append({
                            "player": player_id,
                            "observation": observation,
                            "action": action,
                            "result": info.get("result", "No result recorded")
                        })
                        move_count += 1
                    except Exception as e:
                        logger.error(f"Error during game play: {str(e)}")
                        # If there's an error during gameplay, mark the game as done and move to next
                        done = True
                        rewards = [0, 0]  # Default rewards for failed game
                        current_game_history["outcome"] = "Error"
                        break
                
                rewards = env.close()
                logger.info(f"Game {game_num + 1} completed. Rewards: Player0={rewards[0]}, Player1={rewards[1]}")
                
                # Record game outcome
                if current_game_history["outcome"] != "Error":
                    if rewards[0] > rewards[1]:
                        current_game_history["outcome"] = "Player 0 won"
                    elif rewards[1] > rewards[0]:
                        current_game_history["outcome"] = "Player 1 won"
                    else:
                        current_game_history["outcome"] = "Draw"
                logger.info(f"Game {game_num + 1} outcome: {current_game_history['outcome']}")
                
                # Generate key learnings from the game if it didn't error out
                current_learning = None
                if current_game_history["outcome"] != "Error":
                    try:
                        logger.info("Generating key learnings from the game")
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
                        logger.debug(f"Generated learnings: {current_learning[:200]}...")  # Log first 200 chars of learnings
                        game_learnings[game].append(current_learning)
                    except Exception as e:
                        logger.error(f"Error generating learnings: {str(e)}")
                        current_learning = f"Error generating learnings: {str(e)}"
                
                game_histories[game].append(current_game_history)
                
                # Log results with model names
                result_file = f"results/setting3_{player1_model_name}_results.jsonl"
                with open(result_file, "a") as f:
                    json.dump({
                        "game": game,
                        "game_num": game_num,
                        "rewards": rewards,
                        "history": current_game_history,
                        "learning": current_learning,
                        "player0_model": "qwen2.5-32b-chat",
                        "player1_model": player1_model_name,
                        "error": None if current_game_history["outcome"] != "Error" else str(e)
                    }, f)
                    f.write("\n")
                logger.debug(f"Results saved to {result_file}")
            except Exception as e:
                logger.error(f"Error in game {game_num + 1} of {game}: {str(e)}")
                # Log the error and continue with next game
                result_file = f"results/setting3_{player1_model_name}_results.jsonl"
                with open(result_file, "a") as f:
                    json.dump({
                        "game": game,
                        "game_num": game_num,
                        "rewards": [0, 0],
                        "history": {"moves": [], "outcome": "Error"},
                        "learning": None,
                        "player0_model": "qwen2.5-32b-chat",
                        "player1_model": player1_model_name,
                        "error": str(e)
                    }, f)
                    f.write("\n")
                continue

def main():
    parser = argparse.ArgumentParser(description="Run experiment settings.")
    parser.add_argument("--model-name", type=str, required=True, help="Name to serve the model as")
    parser.add_argument("--games", type=str, nargs="+", help="List of games to evaluate")
    parser.add_argument("--setting", type=int, required=True, help="Setting number to run (1-3)")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    args = parser.parse_args()
    logger.info(f"Starting experiment with arguments: {args}")


    try:
        # Wait for server to start
        # logger.info("Waiting for server to start...")
        # time.sleep(30)

        # Step 2: Run the specified setting
        logger.info(f"Running Setting {args.setting}...")
        if args.setting == 1:
            setting1_basic_evaluation(args.model_name, args.games)
        elif args.setting == 2:
            setting2_player1_with_history(args.model_name, args.games)
        elif args.setting == 3:
            setting3_player0_teacher(args.model_name, args.games)
        else:
            raise ValueError(f"Invalid setting number: {args.setting}")
        
        logger.info(f"Setting {args.setting} completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
    finally:
        # Step 3: Stop the server
        logger.info("Stopping Task : {} , setting : {}".format(args.task, args.setting))

if __name__ == "__main__":
    main() 