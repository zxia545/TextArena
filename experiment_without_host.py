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

# Configure logging with both file and console output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment_without_host.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)



def get_player1_agent(model_name: str):
    """Helper function to create Player 1 agent with specified model"""
    logger.debug(f"Creating Player1 agent with model: {model_name}")
    agent = ta.agents.OpenRouterAgent(
        model_name=model_name,
        api_base="http://localhost:8010/v1",
        api_key="your_api_key_here"
    )
    logger.debug("Player1 agent created successfully")
    return agent

def get_player0_agent():
    """Helper function to create Player 0 agent (Qwen3-32B)"""
    logger.debug("Creating Player0 agent (Qwen3-32B)")
    agent = Qwen3Agent(
        model_name="qwen3-32b",
        api_base="http://localhost:8020/v1",
        api_key="your_api_key_here"
    )
    logger.debug("Player0 agent created successfully")
    return agent

# List of all games to evaluate
ALL_GAMES = [
    "SpellingBee-v0",
    "Poker-v0",
    "SpiteAndMalice-v0",
    "Stratego-v0",
    "Tak-v0",
    "TruthAndDeception-v0",
    "UltimateTicTacToe-v0",
    "WordChains-v0",
    "TicTacToe-v0",
    "Breakthrough-v0",
    "Checkers-v0",
    "KuhnPoker-v0",
    "LetterAuction-v0",
    "MemoryGame-v0",
    "Nim-v0",
    "Othello-v0",
    "PigDice-v0",
    "SimpleBlindAuction-v0",
    "Snake-v0",
    "SecretMafia-v0",
]

# Setting 1: Basic win rate evaluation with TrueSkill
def setting1_basic_evaluation(player1_model_name: str, selected_games: List[str]):
    logger.info(f"Starting Setting 1 evaluation with Player1 model: {player1_model_name}")
    # Initialize agents
    agents = {
        0: get_player0_agent(),
        1: get_player1_agent(player1_model_name)
    }
    logger.debug("Agents initialized successfully")

    # Track results for TrueSkill calculation
    results = {game: [] for game in selected_games}
    logger.debug(f"Initialized results tracking for games: {selected_games}")
    
    # Play each game 10 times
    for game in selected_games:
        logger.info(f"Starting {game} evaluation with {player1_model_name}")
        for game_num in range(10):
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
                player_id, observation = env.get_observation()
                logger.debug(f"Player {player_id}'s turn - Move {move_count + 1}")
                logger.debug(f"Observation: {observation[:200]}...")  # Log first 200 chars of observation
                action = agents[player_id](observation)
                logger.debug(f"Player {player_id} action: {action}")
                done, info = env.step(action=action)
                logger.debug(f"Step info: {info}")
                move_count += 1
            
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
                    "player0_model": "qwen3-32b",
                    "player1_model": player1_model_name
                }, f)
                f.write("\n")
            logger.debug(f"Results saved to {result_file}")

# Setting 2: Player1 with history
def setting2_player1_with_history(player1_model_name: str, selected_games: List[str]):
    logger.info(f"Starting Setting 2 evaluation with Player1 model: {player1_model_name}")
    agents = {
        0: get_player0_agent(),
        1: get_player1_agent(player1_model_name)
    }
    logger.debug("Agents initialized successfully")

    # Store game histories
    game_histories = {game: [] for game in selected_games}
    logger.debug(f"Initialized game histories for games: {selected_games}")
    
    # Play each game 10 times
    for game in selected_games:
        logger.info(f"Starting {game} evaluation with {player1_model_name}")
        for game_num in range(10):
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
                player_id, observation = env.get_observation()
                logger.debug(f"Player {player_id}'s turn - Move {move_count + 1}")
                logger.debug(f"Observation: {observation[:200]}...")  # Log first 200 chars of observation
                action = agents[player_id](observation)
                logger.debug(f"Player {player_id} action: {action}")
                done, info = env.step(action=action)
                logger.debug(f"Step info: {info}")
                
                # Record move with more context
                current_game_history["moves"].append({
                    "player": player_id,
                    "observation": observation,
                    "action": action,
                    "result": info.get("result", "No result recorded")
                })
                move_count += 1
            
            rewards = env.close()
            logger.info(f"Game {game_num + 1} completed. Rewards: Player0={rewards[0]}, Player1={rewards[1]}")
            
            # Record game outcome
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
                    "player0_model": "qwen3-32b",
                    "player1_model": player1_model_name
                }, f)
                f.write("\n")
            logger.debug(f"Results saved to {result_file}")

# Setting 3: Player0 as master teacher for Player1
def setting3_player0_teacher(player1_model_name: str, selected_games: List[str]):
    logger.info(f"Starting Setting 3 evaluation with Player1 model: {player1_model_name}")
    agents = {
        0: get_player0_agent(),
        1: get_player1_agent(player1_model_name)
    }
    logger.debug("Agents initialized successfully")

    # Store game histories and learnings
    game_histories = {game: [] for game in selected_games}
    game_learnings = {game: [] for game in selected_games}
    logger.debug(f"Initialized game histories and learnings for games: {selected_games}")
    
    # Play each game 10 times
    for game in selected_games:
        logger.info(f"Starting {game} evaluation with {player1_model_name}")
        for game_num in range(10):
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
                player_id, observation = env.get_observation()
                logger.debug(f"Player {player_id}'s turn - Move {move_count + 1}")
                logger.debug(f"Observation: {observation[:200]}...")  # Log first 200 chars of observation
                action = agents[player_id](observation)
                logger.debug(f"Player {player_id} action: {action}")
                done, info = env.step(action=action)
                logger.debug(f"Step info: {info}")
                
                # Record move with more context
                current_game_history["moves"].append({
                    "player": player_id,
                    "observation": observation,
                    "action": action,
                    "result": info.get("result", "No result recorded")
                })
                move_count += 1
            
            rewards = env.close()
            logger.info(f"Game {game_num + 1} completed. Rewards: Player0={rewards[0]}, Player1={rewards[1]}")
            
            # Record game outcome
            if rewards[0] > rewards[1]:
                current_game_history["outcome"] = "Player 0 won"
            elif rewards[1] > rewards[0]:
                current_game_history["outcome"] = "Player 1 won"
            else:
                current_game_history["outcome"] = "Draw"
            logger.info(f"Game {game_num + 1} outcome: {current_game_history['outcome']}")
            
            # Generate key learnings from the game
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
                    "player0_model": "qwen3-32b",
                    "player1_model": player1_model_name
                }, f)
                f.write("\n")
            logger.debug(f"Results saved to {result_file}")

def run_all_settings(player1_model_name: str, selected_games: List[str]):
    """Run all settings for a given model"""
    logger.info(f"Starting evaluation run for model: {player1_model_name}")
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    logger.debug("Results directory created/verified")
    
    logger.info(f"Running Setting 2 with {player1_model_name}...")
    setting2_player1_with_history(player1_model_name, selected_games)
    
    logger.info(f"Running Setting 3 with {player1_model_name}...")
    setting3_player0_teacher(player1_model_name, selected_games)
    
    logger.info(f"Completed all settings for model: {player1_model_name}")

def main():
    parser = argparse.ArgumentParser(description="Start vLLM server and run experiment settings.")
    parser.add_argument("--model-name", type=str, required=True, help="Name to serve the model as")
    parser.add_argument("--games", type=str, nargs="+", default=ALL_GAMES, help="List of games to evaluate")
    args = parser.parse_args()
    logger.info(f"Starting experiment with arguments: {args}")

    try:
        logger.info("Starting experiment settings...")
        run_all_settings(args.model_name, args.games)
        logger.info("Experiment settings completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        logger.info("Attempting to run settings again after error...")
        run_all_settings(args.model_name, args.games)

if __name__ == "__main__":
    main() 