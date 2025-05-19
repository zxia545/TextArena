import textarena as ta
import json
from typing import Dict, List, Tuple, Optional
import os
import argparse
import subprocess
import sys
import time
from this_utils import start_vllm_server, stop_vllm_server
from textarena.agents.basic_agents import Qwen3Agent
import logging
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Lock
import re

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename='experiment.log')
logger = logging.getLogger(__name__)

# # Selected games for evaluation
# SELECTED_GAMES = [
#     "TicTacToe-v0",
#     "Poker-v0",
#     "Stratego-v0",
#     "TruthAndDeception-v0",
#     "UltimateTicTacToe-v0",
#     "Checkers-v0",
#     "Othello-v0",
#     "SecretMafia-v0"
# ]

# SELECTED_GAMES =[
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
#     "WildTicTacToe-v0",
#     "ReverseTicTacToe-v0",
#     "RandomizedTicTacToe-v0",
#     "QuantumTicTacToe-v0",
# ]


SELECTED_GAMES = [
    "TicTacToe-v0",
    "Poker-v0",
    "Stratego-v0",
    "TruthAndDeception-v0",
    "UltimateTicTacToe-v0",
    "Checkers-v0",
    "Othello-v0",
    # "SecretMafia-v0"
]


def get_player1_agent(model_name: str):
    """Helper function to create Player 1 agent with specified model"""
    return ta.agents.OpenRouterAgent(
        model_name=model_name,
        api_base="http://localhost:8010/v1",
        api_key="your_api_key_here",
        timeout=70
    )

def get_player0_agent():
    """Helper function to create Player 0 agent (Qwen2.5-32B)"""
    return ta.agents.OpenRouterAgent(
        model_name="qwen2.5-32b-chat",
        api_base="http://localhost:8020/v1",
        api_key="your_api_key_here",
        timeout=70
    )


# Setting 1: Basic win rate evaluation with TrueSkill
def setting1_basic_evaluation(player1_model_name: str, selected_games: List[str], output_file: str, num_rounds: int):
    logger.info(f"Starting Setting 1 evaluation with Player1 model: {player1_model_name}")
    # Initialize agents
    agents = {
        0: get_player0_agent(),
        1: get_player1_agent(player1_model_name)
    }
    logger.info("Agents initialized successfully")

    # Track results for TrueSkill calculation
    results = {game: [] for game in selected_games}
    
    # Play each game num_rounds times
    for game in selected_games:
        logger.info(f"Starting {game} evaluation with {player1_model_name}")
        for game_num in range(num_rounds):
            try:
                logger.info(f"Starting game {game_num + 1}/{num_rounds} for {game}")
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
                with open(output_file, "a") as f:
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
                logger.debug(f"Results saved to {output_file}")
            except Exception as e:
                logger.error(f"Error in game {game_num + 1} of {game}: {str(e)}")
                # Log the error and continue with next game
                with open(output_file, "a") as f:
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
def setting2_player1_with_history(player1_model_name: str, selected_games: List[str], output_file: str, num_rounds: int):
    logger.info(f"Starting Setting 2 evaluation with Player1 model: {player1_model_name}")
    agents = {
        0: get_player0_agent(),
        1: get_player1_agent(player1_model_name)
    }
    logger.info("Agents initialized successfully")

    # Store game histories and learnings
    game_histories = {game: [] for game in selected_games}
    game_learnings = {game: [] for game in selected_games}
    
    # Play each game num_rounds times
    for game in selected_games:
        logger.info(f"Starting {game} evaluation with {player1_model_name}")
        for game_num in range(num_rounds):
            try:
                logger.info(f"Starting game {game_num + 1}/{num_rounds} for {game}")
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
                        
                        current_game_history["moves"].append({
                            "player": player_id,
                            "observation": observation,
                            "action": action
                        })
                        move_count += 1
                    except Exception as e:
                        logger.error(f"Error during game play: {str(e)}")
                        done = True
                        rewards = [0, 0]
                        current_game_history["outcome"] = "Error"
                        break
                
                rewards = env.close()
                logger.info(f"Game {game_num + 1} completed. Rewards: Player0={rewards[0]}, Player1={rewards[1]}")
                
                # Record game outcome
                if current_game_history["outcome"] != "Error":
                    if rewards[0] > rewards[1]:
                        current_game_history["outcome"] = "Player 0 won"
                        winning_player = 0
                    elif rewards[1] > rewards[0]:
                        current_game_history["outcome"] = "Player 1 won"
                        winning_player = 1
                    else:
                        current_game_history["outcome"] = "Draw"
                        winning_player = None
                logger.info(f"Game {game_num + 1} outcome: {current_game_history['outcome']}")
                
                # Generate key learnings from the game if it didn't error out
                current_learning = None
                if current_game_history["outcome"] != "Error" and current_game_history["moves"]:
                    try:
                        logger.info("Generating key learnings from the game")
                        # Get the last move
                        last_move = current_game_history["moves"][-1]
                        
                        learning_prompt = (
                            f"Please analyze this game and provide key learnings for your future play.\n"
                            f"Game outcome: {current_game_history['outcome']}\n"
                            f"Last move details:\n"
                            f"Player: {last_move['player']}\n"
                            f"Observation: {last_move['observation']}\n"
                            f"Action taken: {last_move['action']}\n\n"
                            f"Provide learnings in this format:\n"
                            f"1. Strategic principles to follow\n"
                            f"2. Specific moves to consider\n"
                            f"3. Moves to avoid\n"
                            f"4. Key patterns to watch for"
                        )
                        current_learning = agents[1](learning_prompt)
                        logger.debug(f"Generated learnings: {current_learning[:200]}...")
                        game_learnings[game].append(current_learning)
                    except Exception as e:
                        logger.error(f"Error generating learnings: {str(e)}")
                        current_learning = f"Error generating learnings: {str(e)}"
                
                game_histories[game].append(current_game_history)
                
                # Log results with model names
                with open(output_file, "a") as f:
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
                logger.debug(f"Results saved to {output_file}")
            except Exception as e:
                logger.error(f"Error in game {game_num + 1} of {game}: {str(e)}")
                # Log the error and continue with next game
                with open(output_file, "a") as f:
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

# Setting 3: Player0 as master teacher for Player1
def setting3_player0_teacher(player1_model_name: str, selected_games: List[str], output_file: str, num_rounds: int):
    logger.info(f"Starting Setting 3 evaluation with Player1 model: {player1_model_name}")
    agents = {
        0: get_player0_agent(),
        1: get_player1_agent(player1_model_name)
    }
    logger.info("Agents initialized successfully")

    # Store game histories and learnings
    game_histories = {game: [] for game in selected_games}
    game_learnings = {game: [] for game in selected_games}
    
    # Play each game num_rounds times
    for game in selected_games:
        logger.info(f"Starting {game} evaluation with {player1_model_name}")
        for game_num in range(num_rounds):
            try:
                logger.info(f"Starting game {game_num + 1}/{num_rounds} for {game}")
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
                        
                        current_game_history["moves"].append({
                            "player": player_id,
                            "observation": observation,
                            "action": action
                        })
                        move_count += 1
                    except Exception as e:
                        logger.error(f"Error during game play: {str(e)}")
                        done = True
                        rewards = [0, 0]
                        current_game_history["outcome"] = "Error"
                        break
                
                rewards = env.close()
                logger.info(f"Game {game_num + 1} completed. Rewards: Player0={rewards[0]}, Player1={rewards[1]}")
                
                # Record game outcome
                if current_game_history["outcome"] != "Error":
                    if rewards[0] > rewards[1]:
                        current_game_history["outcome"] = "Player 0 won"
                        winning_player = 0
                    elif rewards[1] > rewards[0]:
                        current_game_history["outcome"] = "Player 1 won"
                        winning_player = 1
                    else:
                        current_game_history["outcome"] = "Draw"
                        winning_player = None
                logger.info(f"Game {game_num + 1} outcome: {current_game_history['outcome']}")
                
                # Generate key learnings from the game if it didn't error out
                current_learning = None
                if current_game_history["outcome"] != "Error" and current_game_history["moves"]:
                    try:
                        logger.info("Generating key learnings from the game")
                        # Get the last move
                        last_move = current_game_history["moves"][-1]
                        
                        learning_prompt = (
                            f"As a master teacher, analyze this game and provide key learnings for Player1.\n"
                            f"Game outcome: {current_game_history['outcome']}\n"
                            f"Last move details:\n"
                            f"Player: {last_move['player']}\n"
                            f"Observation: {last_move['observation']}\n"
                            f"Action taken: {last_move['action']}\n\n"
                            f"Provide learnings in this format:\n"
                            f"1. Strategic principles to follow\n"
                            f"2. Specific moves to consider\n"
                            f"3. Moves to avoid\n"
                            f"4. Key patterns to watch for"
                        )
                        current_learning = agents[0](learning_prompt)
                        logger.debug(f"Generated learnings: {current_learning[:200]}...")
                        game_learnings[game].append(current_learning)
                    except Exception as e:
                        logger.error(f"Error generating learnings: {str(e)}")
                        current_learning = f"Error generating learnings: {str(e)}"
                
                game_histories[game].append(current_game_history)
                
                # Log results with model names
                with open(output_file, "a") as f:
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
                logger.debug(f"Results saved to {output_file}")
            except Exception as e:
                logger.error(f"Error in game {game_num + 1} of {game}: {str(e)}")
                # Log the error and continue with next game
                with open(output_file, "a") as f:
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

def setting4_combined_learning(player1_model_name: str, selected_games: List[str], output_file: str, num_rounds: int):
    logger.info(f"Starting Setting 4 evaluation with Player1 model: {player1_model_name}")
    agents = {
        0: get_player0_agent(),
        1: get_player1_agent(player1_model_name)
    }
    logger.info("Agents initialized successfully")

    # Store game histories, advice, learnings and scores
    game_histories = {game: [] for game in selected_games}
    player0_advice = {game: [] for game in selected_games}
    player1_learnings = {game: [] for game in selected_games}
    game_scores = {game: [] for game in selected_games}
    
    def get_top_histories(game: str, n: int = 3) -> List[Dict]:
        """Get the top N highest scored game histories"""
        if not game_scores[game]:
            return []
        
        # Sort games by score
        sorted_games = sorted(
            range(len(game_scores[game])),
            key=lambda i: game_scores[game][i],
            reverse=True
        )
        
        # Get top N games
        top_indices = sorted_games[:n]
        return [game_histories[game][i] for i in top_indices]
    
    def create_history_prompt(game: str) -> str:
        """Create a prompt with the top game histories and their learnings"""
        top_histories = get_top_histories(game)
        if not top_histories:
            return ""
        
        prompt = "Previous successful games and their learnings:\n\n"
        for i, history in enumerate(top_histories, 1):
            prompt += f"Game {i}:\n"
            if history["moves"]:
                last_move = history["moves"][-1]
                prompt += f"Last move:\n"
                prompt += f"Player: {last_move['player']}\n"
                prompt += f"Observation: {last_move['observation']}\n"
                prompt += f"Action: {last_move['action']}\n"
            prompt += f"Outcome: {history['outcome']}\n"
            prompt += f"Score: {game_scores[game][i-1]}\n"
            prompt += f"Learning: {player1_learnings[game][i-1]}\n\n"
        return prompt
    
    def extract_score(response: str) -> Optional[int]:
        """Extract score from LLM response using regex"""
        match = re.search(r'(\d+)/10', response)
        if match:
            return int(match.group(1))
        return None
    
    def get_score_with_retry(prompt: str, max_retries: int = 3) -> Optional[int]:
        """Get score from Player1 with retry mechanism"""
        for attempt in range(max_retries):
            try:
                response = agents[1](prompt)
                score = extract_score(response)
                if score is not None:
                    return score
            except Exception as e:
                logger.error(f"Error getting score (attempt {attempt + 1}): {str(e)}")
        return None
    
    # Play each game num_rounds times
    for game in selected_games:
        logger.info(f"Starting {game} evaluation with {player1_model_name}")
        for game_num in range(num_rounds):
            try:
                logger.info(f"Starting game {game_num + 1}/{num_rounds} for {game}")
                env = ta.make(env_id=game)
                env = ta.wrappers.LLMObservationWrapper(env=env)
                env = ta.wrappers.SimpleRenderWrapper(
                    env=env,
                    player_names={0: "Player0", 1: "Player1"},
                )
                logger.debug(f"Environment created and configured for {game}")

                # Add previous successful games to Player1's prompt
                if game_num > 0:
                    logger.info("Adding previous successful games to Player1's prompt")
                    history_prompt = create_history_prompt(game)
                    if history_prompt:
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
                        
                        current_game_history["moves"].append({
                            "player": player_id,
                            "observation": observation,
                            "action": action
                        })
                        move_count += 1
                    except Exception as e:
                        logger.error(f"Error during game play: {str(e)}")
                        done = True
                        rewards = [0, 0]
                        current_game_history["outcome"] = "Error"
                        break
                
                rewards = env.close()
                logger.info(f"Game {game_num + 1} completed. Rewards: Player0={rewards[0]}, Player1={rewards[1]}")
                
                # Record game outcome
                if current_game_history["outcome"] != "Error":
                    if rewards[0] > rewards[1]:
                        current_game_history["outcome"] = "Player 0 won"
                        winning_player = 0
                    elif rewards[1] > rewards[0]:
                        current_game_history["outcome"] = "Player 1 won"
                        winning_player = 1
                    else:
                        current_game_history["outcome"] = "Draw"
                        winning_player = None
                logger.info(f"Game {game_num + 1} outcome: {current_game_history['outcome']}")
                
                # Generate advice and learnings if game didn't error out
                current_advice = None
                current_learning = None
                current_score = None
                
                if current_game_history["outcome"] != "Error" and current_game_history["moves"]:
                    try:
                        logger.info("Generating advice and learnings")
                        # Get the last move
                        last_move = current_game_history["moves"][-1]
                        
                        analysis_prompt = (
                            f"Analyze this game and provide strategic advice.\n"
                            f"Game outcome: {current_game_history['outcome']}\n"
                            f"Last move details:\n"
                            f"Player: {last_move['player']}\n"
                            f"Observation: {last_move['observation']}\n"
                            f"Action taken: {last_move['action']}\n\n"
                            f"Provide your analysis in this format:\n"
                            f"1. Strategic principles to follow\n"
                            f"2. Specific moves to consider\n"
                            f"3. Moves to avoid\n"
                            f"4. Key patterns to watch for\n\n"
                            f"Then rate the quality of this game on a scale of 0-10, where:\n"
                            f"0-2: Poor - Basic mistakes, no strategy\n"
                            f"3-4: Fair - Some good moves but inconsistent\n"
                            f"5-6: Good - Solid play with clear strategy\n"
                            f"7-8: Very Good - Strong tactical play\n"
                            f"9-10: Excellent - Masterful play with perfect execution\n"
                            f"End your response with 'Score: X/10' where X is your rating."
                        )
                        
                        # Get advice from Player0
                        current_advice = agents[0](analysis_prompt)
                        player0_advice[game].append(current_advice)
                        
                        # Get learnings and score from Player1
                        current_learning = agents[1](analysis_prompt)
                        player1_learnings[game].append(current_learning)
                        
                        # Extract score
                        current_score = extract_score(current_learning)
                        if current_score is not None:
                            game_scores[game].append(current_score)
                            logger.info(f"Game {game_num + 1} scored {current_score}/10")
                        else:
                            logger.warning(f"Could not extract score from Player1's response")
                            game_scores[game].append(0)
                    except Exception as e:
                        logger.error(f"Error generating advice/learnings: {str(e)}")
                        current_advice = f"Error generating advice: {str(e)}"
                        current_learning = f"Error generating learnings: {str(e)}"
                        game_scores[game].append(0)
                
                game_histories[game].append(current_game_history)
                
                # Log results with model names
                with open(output_file, "a") as f:
                    json.dump({
                        "game": game,
                        "game_num": game_num,
                        "rewards": rewards,
                        "history": current_game_history,
                        "player0_advice": current_advice,
                        "player1_learning": current_learning,
                        "score": current_score,
                        "player0_model": "qwen2.5-32b-chat",
                        "player1_model": player1_model_name,
                        "error": None if current_game_history["outcome"] != "Error" else str(e)
                    }, f)
                    f.write("\n")
                logger.debug(f"Results saved to {output_file}")
            except Exception as e:
                logger.error(f"Error in game {game_num + 1} of {game}: {str(e)}")
                # Log the error and continue with next game
                with open(output_file, "a") as f:
                    json.dump({
                        "game": game,
                        "game_num": game_num,
                        "rewards": [0, 0],
                        "history": {"moves": [], "outcome": "Error"},
                        "player0_advice": None,
                        "player1_learning": None,
                        "score": None,
                        "player0_model": "qwen2.5-32b-chat",
                        "player1_model": player1_model_name,
                        "error": str(e)
                    }, f)
                    f.write("\n")
                continue

def run_single_setting(setting_num: int, game: str, model_name: str, output_file: str, num_rounds: int):
    """Run a single setting for a specific game"""
    logger.info(f"Starting Setting {setting_num} for game {game} with model {model_name}")
    
    try:
        if setting_num == 1:
            setting1_basic_evaluation(model_name, [game], output_file, num_rounds)
        elif setting_num == 2:
            setting2_player1_with_history(model_name, [game], output_file, num_rounds)
        elif setting_num == 3:
            setting3_player0_teacher(model_name, [game], output_file, num_rounds)
        elif setting_num == 4:
            setting4_combined_learning(model_name, [game], output_file, num_rounds)
        else:
            raise ValueError(f"Invalid setting number: {setting_num}")
        
        logger.info(f"Setting {setting_num} completed for game {game}")
        return True
    except Exception as e:
        logger.error(f"Error in Setting {setting_num} for game {game}: {str(e)}")
        return False

def run_parallel_evaluation(model_name: str, output_dir: str, max_concurrent: int = 9, num_rounds: int = 10, selected_game_mode_list=[]):
    """Run all games and settings in parallel with a maximum of concurrent tasks"""
    logger.info(f"Starting parallel evaluation with max {max_concurrent} concurrent tasks")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create task queue with all combinations
    task_queue = Queue()
    for game in SELECTED_GAMES:
        for setting in selected_game_mode_list:
            output_file = os.path.join(output_dir, f"setting{setting}_{game}_{model_name}_results.jsonl")
            task_queue.put((setting, game, output_file, num_rounds))
    
    # Track completed and failed tasks
    completed_tasks = set()
    failed_tasks = []
    task_lock = Lock()
    
    def worker():
        while True:
            try:
                # Get next task
                setting, game, output_file, num_rounds = task_queue.get_nowait()
                task_id = f"{setting}_{game}"
                
                # Run the task
                success = run_single_setting(setting, game, model_name, output_file, num_rounds)
                
                # Update task status
                with task_lock:
                    if success:
                        completed_tasks.add(task_id)
                        logger.info(f"Task {task_id} completed successfully")
                    else:
                        failed_tasks.append((setting, game))
                        logger.error(f"Task {task_id} failed")
                
                task_queue.task_done()
            except Queue.Empty:
                break
            except Exception as e:
                logger.error(f"Worker error: {str(e)}")
                task_queue.task_done()
    
    # Start workers
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        # Submit initial batch of workers
        futures = [executor.submit(worker) for _ in range(max_concurrent)]
        
        # Wait for all tasks to complete
        task_queue.join()
    
    # Log final status
    logger.info(f"All tasks completed. Success: {len(completed_tasks)}, Failed: {len(failed_tasks)}")
    
    # Retry failed tasks once
    if failed_tasks:
        logger.info(f"Retrying {len(failed_tasks)} failed tasks...")
        for setting, game in failed_tasks:
            output_file = os.path.join(output_dir, f"setting{setting}_{game}_{model_name}_retry_results.jsonl")
            success = run_single_setting(setting, game, model_name, output_file, num_rounds)
            if success:
                logger.info(f"Retry successful for Setting {setting}, Game {game}")
            else:
                logger.error(f"Retry failed for Setting {setting}, Game {game}")

def main():
    parser = argparse.ArgumentParser(description="Run parallel evaluation of games and settings")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    parser.add_argument("--port", type=int, default=8010, help="Port to serve the model on")
    parser.add_argument("--gpu", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--model-name", type=str, required=True, help="Name to serve the model as")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory path")
    parser.add_argument("--max-concurrent", type=int, default=9, help="Maximum number of concurrent tasks")
    parser.add_argument("--num-rounds", type=int, default=10, help="Number of rounds to play for each game")
    parser.add_argument("--select_mode", type=str, default="1,2,3,4", help="Number of games to run")
    # parser.add_argument("--tasks", type=str, default="all", help="Tasks to run")
    args = parser.parse_args()
    
    logger.info(f"Starting experiment with arguments: {args}")
    
    logger.info(f"Starting vLLM server for {args.model_name}...")
    server_proc = start_vllm_server(args.model_path, args.model_name, port=args.port, gpu=args.gpu)
    selected_game_mode_list = args.select_mode.split(",")
    try:
        run_parallel_evaluation(args.model_name, args.output_dir, args.max_concurrent, args.num_rounds, selected_game_mode_list)
        logger.info("All evaluations completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
    finally:
        stop_vllm_server(server_proc)

if __name__ == "__main__":
    main() 