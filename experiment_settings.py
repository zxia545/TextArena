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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=f'logs/experiment_{time.strftime("%Y%m%d_%H%M%S")}.log')
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

# "Checkers-v0,Stratego-v0,TicTacToe-v0,TruthAndDeception-v0,SpellingBee-v0,SpiteAndMalice-v0,Tak-v0,WordChains-v0"
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
    agent = ta.agents.OpenRouterAgent(
        model_name=model_name,
        api_base="http://localhost:8010/v1",
        api_key="your_api_key_here",
        timeout=120
    )
    # Wrap the agent call to provide better error handling
    original_call = agent.__call__
    def safe_call(observation: str) -> str:
        try:
            response = original_call(observation)
            if response is None:
                logger.error(f"Agent {model_name} returned None response")
                return "Error: No response generated"
            return response
        except Exception as e:
            logger.error(f"Error in {model_name} agent call: {type(e).__name__}: {str(e)}")
            return f"Error: {type(e).__name__}: {str(e)}"
    
    agent.__call__ = safe_call
    return agent

def get_player0_agent():
    """Helper function to create Player 0 agent (Qwen2.5-32B)"""
    agent = ta.agents.OpenRouterAgent(
        model_name="qwen2.5-32b-chat",
        api_base="http://localhost:8020/v1",
        api_key="your_api_key_here",
        timeout=120
    )
    # Wrap the agent call to provide better error handling
    original_call = agent.__call__
    def safe_call(observation: str) -> str:
        try:
            response = original_call(observation)
            if response is None:
                logger.error("Player0 agent returned None response")
                return "Error: No response generated"
            return response
        except Exception as e:
            logger.error(f"Error in Player0 agent call: {type(e).__name__}: {str(e)}")
            return f"Error: {type(e).__name__}: {str(e)}"
    
    agent.__call__ = safe_call
    return agent


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

def get_game_summary(game: str, agent, env) -> str:
    """Generate or load a summary of the game rules with strategic advice for winning
    
    Args:
        game: The game ID
        agent: The agent to use for generating the summary
        env: The game environment
        
    Returns:
        A string containing the game summary with winning strategies
    """
    # Define the directory and file path
    summary_dir = "environment_summary"
    summary_file = os.path.join(summary_dir, f"{game}.jsonl")
    
    # Check if summary already exists
    if os.path.exists(summary_file):
        logger.info(f"Loading existing game summary for {game}")
        try:
            with open(summary_file, 'r') as f:
                summary_data = json.load(f)
                return summary_data.get("summary", "")
        except Exception as e:
            logger.error(f"Error loading game summary: {str(e)}")
            # If we fail to load, we'll generate a new one
    
    # If we get here, we need to generate a summary
    logger.info(f"Generating new game summary for {game}")
    
    # Reset the environment to get the first observation
    env.reset(num_players=2)  # Assuming 2 players for all games
    player_id, observation = env.get_observation()
    
    # Save original system prompt
    original_system_prompt = agent.system_prompt
    
    # Set new system prompt for summary generation with strategic advice
    summary_prompt = "You are an expert game strategist with deep knowledge of game theory and optimal play. Your task is to provide concise, actionable strategic advice that will help a player win. Focus on identifying winning patterns, key decision points, and optimal strategies."
    agent.system_prompt = summary_prompt
    
    # Generate the summary with strategic advice
    prompt = (
        f"For this game '{game}', provide very brief winning strategies based on this initial observation. Don't explain rules in detail.\n\n"
        f"WINNING STRATEGIES:\n"
        f"- Top 3-5 strategic principles that lead to victory\n"
        f"- Best opening moves or early game tactics\n"
        f"- Key patterns to recognize during gameplay\n" 
        f"- Critical mistakes to avoid\n\n"
        f"Game observation: {observation}\n\n"
        f"Keep your response concise and focused on practical advice that will maximize winning chances."
    )
    
    try:
        summary = agent(prompt)
        
        # Create directory if it doesn't exist
        os.makedirs(summary_dir, exist_ok=True)
        
        # Save the summary
        with open(summary_file, 'w') as f:
            json.dump({"game": game, "summary": summary}, f)
            
        logger.info(f"Saved game summary for {game}")
    except Exception as e:
        logger.error(f"Error generating game summary: {str(e)}")
        summary = f"Error generating summary: {str(e)}"
    
    # Restore original system prompt
    agent.system_prompt = original_system_prompt
    
    # Reset env to clean state before returning
    env.close()
    
    return summary

def setting4_combined_learning(player1_model_name: str, selected_games: List[str], output_file: str, num_rounds: int, history_limit: Optional[int] = None):
    logger.info(f"Starting Setting 4 evaluation with Player1 model: {player1_model_name}")
    agents = {
        0: get_player0_agent(),
        1: get_player1_agent(player1_model_name)
    }
    logger.info("Agents initialized successfully")

    # Store game histories, advice, learnings and scores
    game_histories = {game: [] for game in selected_games}
    player0_advice = {game: [] for game in selected_games}
    player0_scores = {game: [] for game in selected_games}
    player1_advice = {game: [] for game in selected_games}
    game_scores = {game: [] for game in selected_games}
    history_prompts = {game: [] for game in selected_games}
    game_summaries = {}  # Store game summaries
    
    
    def create_history_prompt(game: str) -> str:
        """Create a prompt with the top game histories and their learnings from before the history limit"""
        if not game_scores[game]:
            return ""
            
        # Only consider games before history limit
        valid_indices = range(len(game_scores[game]))
        if history_limit is not None:
            valid_indices = range(min(history_limit, len(game_scores[game])))
        
        # Get sorted indices of top games
        top_indices = sorted(
            valid_indices,
            key=lambda i: game_scores[game][i],
            reverse=True
        )[:3] # Get top 3 games
        
        if not top_indices:
            return ""
        
        # Collect advice from top games for both players
        player0_advice_list = []
        player1_advice_list = []
        
        for idx in top_indices:
            p0_advice = player0_advice[game][idx]
            p1_advice = player1_advice[game][idx]
            
            # Only include valid advice
            if p0_advice and not "Error generating advice" in p0_advice:
                player0_advice_list.append(p0_advice)
            
            if p1_advice and not "Error generating advice" in p1_advice:
                player1_advice_list.append(p1_advice)
        
        # If we have no valid advice, return empty prompt
        if not player0_advice_list and not player1_advice_list:
            return ""
            
        # Have each agent combine their respective advice
        combined_p0_advice = ""
        combined_p1_advice = ""
        
        try:
            # Combine Player 0's advice if any exists
            if player0_advice_list:
                # Save original system prompt
                original_p0_prompt = agents[0].system_prompt
                try:
                    # Set system prompt for combining advice
                    agents[0].system_prompt = "You are an expert game strategist. Synthesize these previous pieces of advice into a single coherent strategic guide. Focus on common themes, contradictory advice to resolve, and the most important strategic principles. Keep your response under 300 words."
                    
                    # Create prompt for combining advice
                    p0_combine_prompt = f"You've provided the following advice about the game {game} from your previous analysis. Combine and synthesize this advice into a single coherent strategic guide:\n\n"
                    for i, advice in enumerate(player0_advice_list, 1):
                        p0_combine_prompt += f"Advice #{i}:\n{advice}\n\n"
                    p0_combine_prompt += "Synthesize the above advice into a single coherent strategic guide with your most important recommendations. Keep your response under 300 words."
                    
                    # Get combined advice
                    combined_p0_advice = agents[0](p0_combine_prompt)
                finally:
                    # Restore original system prompt
                    agents[0].system_prompt = original_p0_prompt
            
            # Combine Player 1's advice if any exists
            if player1_advice_list:
                # Save original system prompt
                original_p1_prompt = agents[1].system_prompt
                try:
                    # Set system prompt for combining advice
                    agents[1].system_prompt = "You are analyzing your own gameplay across multiple matches. Synthesize your previous insights into a single coherent learning summary. Focus on the most important patterns and strategies you've identified. Keep your response under 300 words."
                    
                    # Create prompt for combining advice
                    p1_combine_prompt = f"You've provided the following self-analysis about your gameplay in {game}. Combine and synthesize this analysis into a single coherent learning summary:\n\n"
                    for i, advice in enumerate(player1_advice_list, 1):
                        p1_combine_prompt += f"Analysis #{i}:\n{advice}\n\n"
                    p1_combine_prompt += "Synthesize the above self-analysis into a single coherent learning summary with your most important insights. Keep your response under 300 words."
                    
                    # Get combined advice
                    combined_p1_advice = agents[1](p1_combine_prompt)
                finally:
                    # Restore original system prompt
                    agents[1].system_prompt = original_p1_prompt
        
        except Exception as e:
            logger.error(f"Error combining advice: {str(e)}")
            # Fall back to the original implementation in case of error
            prompt = "Review of previous successful games advice and learnings to inform your strategy:\n\n"
            for i, idx in enumerate(top_indices, 1):
                prompt += f"Game {i} Key Learnings and Advice:\n"
                advice0 = player0_advice[game][idx]
                advice1 = player1_advice[game][idx]
                if advice0 and not "Error generating advice" in advice0:
                    prompt += f"  - Expert Advice: {advice0}\n"
                if advice1 and not "Error generating advice" in advice1:
                    prompt += f"  - Player Analysis: {advice1}\n"
                if not (advice0 and not "Error generating advice" in advice0) and not (advice1 and not "Error generating advice" in advice1):
                    prompt += f"  No specific learnings or advice recorded for Game {i}.\n"
                prompt += "\n"
            return prompt
            
        # Create the final prompt with combined advice
        prompt = "Review of previous successful games advice and learnings to inform your strategy:\n\n"
        
        if combined_p0_advice:
            prompt += f"Expert Strategic Advice:\n{combined_p0_advice}\n\n"
        
        if combined_p1_advice:
            prompt += f"Self-Analysis and Learning:\n{combined_p1_advice}\n\n"
            
        return prompt
    
    def extract_score(response: str) -> Optional[int]:
        """Extract score from LLM response using regex"""
        match = re.search(r'(\d+)/10', response)
        if match:
            return int(match.group(1))
        return None
    
    # Play each game num_rounds times
    for game in selected_games:
        logger.info(f"Starting {game} evaluation with {player1_model_name}")
        
        # Create temporary environment to get game summary
        try:
            temp_env = ta.make(env_id=game)
            temp_env = ta.wrappers.LLMObservationWrapper(env=temp_env)
            temp_env = ta.wrappers.SimpleRenderWrapper(
                env=temp_env,
                player_names={0: "Player0", 1: "Player1"},
            )
            # Get or generate game summary
            game_summary = get_game_summary(game, agents[0], temp_env)
            game_summaries[game] = game_summary
            logger.info(f"Game summary for {game}: {game_summary[:100]}...")
        except Exception as e:
            logger.error(f"Error getting game summary for {game}: {str(e)}")
            game_summaries[game] = f"Error getting game summary: {str(e)}"
        
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
                history_prompt = ""
                prompt_error = None
                if game_num > 0:
                    logger.info("Adding previous successful games to Player1's prompt")
                    try:
                        history_prompt = create_history_prompt(game)
                        if history_prompt:
                            # Store original prompt before adding to system prompt
                            original_prompt = agents[1].system_prompt
                            try:
                                # Add game summary at the beginning if available
                                if game in game_summaries and game_summaries[game] and not "Error" in game_summaries[game]:
                                    combined_prompt = f"GAME ANALYSIS AND WINNING STRATEGIES:\n{game_summaries[game]}\n\nREMEMBER: Apply these strategic principles consistently to maximize your chances of winning.\n\n{history_prompt}\n{original_prompt}"
                                else:
                                    combined_prompt = f"{history_prompt}\n{original_prompt}"
                                    
                                agents[1].system_prompt = combined_prompt
                                logger.debug("History prompt added to Player1's system prompt")
                            except Exception as prompt_error:
                                logger.error(f"Error when setting system prompt: {str(prompt_error)}")
                                agents[1].system_prompt = original_prompt  # Reset to original
                                prompt_error = {
                                    "error": f"{type(prompt_error).__name__}: {str(prompt_error)}",
                                    "prompt": history_prompt,
                                    "prompt_length": len(history_prompt) if history_prompt else 0
                                }
                                raise prompt_error
                    except Exception as e:
                        prompt_error = {
                            "error": f"{type(e).__name__}: {str(e)}",
                            "prompt": history_prompt,
                            "prompt_length": len(history_prompt) if history_prompt else 0
                        }
                        logger.error(f"Error adding history prompt: {str(e)}")
                        # Try to save the problematic prompt to a file for inspection
                        try:
                            with open(f"{output_file}.error_prompt.txt", "w") as f:
                                f.write(history_prompt)
                        except:
                            logger.error("Could not save error prompt to file")
                else:
                    # For the first game, just add the game summary to the prompt
                    if game in game_summaries and game_summaries[game] and not "Error" in game_summaries[game]:
                        original_prompt = agents[1].system_prompt
                        agents[1].system_prompt = f"GAME ANALYSIS AND WINNING STRATEGIES:\n{game_summaries[game]}\n\nREMEMBER: Apply these strategic principles consistently to maximize your chances of winning.\n\n{original_prompt}"
                
                # Store the history prompt for this game
                history_prompts[game].append(history_prompt)

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
                        try:
                            action = agents[player_id](observation)
                            logger.debug(f"Player {player_id} action: {action}")
                        except Exception as agent_error:
                            logger.error(f"Error with agent {player_id} ({type(agents[player_id]).__name__}): {str(agent_error)}")
                            error_details = {
                                "agent_id": player_id,
                                "agent_type": type(agents[player_id]).__name__,
                                "error_type": type(agent_error).__name__,
                                "error_msg": str(agent_error),
                                "agent_system_prompt": agents[player_id].system_prompt,
                                "observation": observation
                            }
                            with open(output_file + ".agent_errors.json", "a") as error_file:
                                json.dump(error_details, error_file)
                                error_file.write("\n")
                            raise agent_error
                        
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
                
                # Initialize variables for advice, scores, and errors
                current_player0_advice = None
                current_player1_advice = None
                current_player0_score = None
                current_score = None
                analysis_prompt_error = None
                scoring_prompt_error = None
                
                # Only generate analysis if we haven't reached the history limit
                if history_limit is None or game_num < history_limit:
                    if current_game_history["outcome"] != "Error" and current_game_history["moves"]:
                        try:
                            logger.info("Generating advice and scores")
                            # Get the last move
                            last_move = current_game_history["moves"][-1]
                            
                            # 1. FIRST TASK: Generate advice from both players
                            advice_prompt = (
                                f"Analyze this game and provide strategic advice.\n"
                                f"Game outcome: {current_game_history['outcome']}\n"
                                f"Final observation: {last_move['observation']}\n"
                                f"Final action: {last_move['action']}\n\n"
                                f"Provide your analysis in this format:\n"
                                f"1. Strategic principles to follow\n"
                                f"2. Specific moves to consider\n"
                                f"3. Moves to avoid\n"
                                f"4. Key patterns to watch for\n\n"
                                f"IMPORTANT: Keep your response concise and focused. Limit your total response to maximum 300 words. Prioritize quality strategic insights over verbose explanations."
                            )
                            
                            # Save original system prompts
                            original_player0_prompt = agents[0].system_prompt
                            original_player1_prompt = agents[1].system_prompt
                            
                            try:
                                # Update system prompts for advice task
                                agents[0].system_prompt = "You are an expert game analyst. Your task is to provide detailed strategic advice for the game based on the provided information. Keep your advice concise and actionable, under 300 words total."
                                agents[1].system_prompt = "You are a player analyzing your own gameplay. Your task is to identify strengths, weaknesses, and provide guidance for future games. Keep your analysis brief and focused, under 300 words total."
                                
                                # Get advice from both players
                                try:
                                    current_player0_advice = agents[0](advice_prompt)
                                    if current_player0_advice is None or "Error:" in current_player0_advice:
                                        logger.error(f"Error getting advice from Player 0: {current_player0_advice}")
                                        current_player0_advice = "Error generating advice"
                                    player0_advice[game].append(current_player0_advice)
                                except Exception as e:
                                    logger.error(f"Exception getting advice from Player 0: {type(e).__name__}: {str(e)}")
                                    current_player0_advice = f"Error: {type(e).__name__}: {str(e)}"
                                    player0_advice[game].append(current_player0_advice)
                                
                                try:
                                    current_player1_advice = agents[1](advice_prompt)
                                    if current_player1_advice is None or "Error:" in current_player1_advice:
                                        logger.error(f"Error getting advice from Player 1: {current_player1_advice}")
                                        current_player1_advice = "Error generating advice"
                                    player1_advice[game].append(current_player1_advice)
                                except Exception as e:
                                    logger.error(f"Exception getting advice from Player 1: {type(e).__name__}: {str(e)}")
                                    current_player1_advice = f"Error: {type(e).__name__}: {str(e)}"
                                    player1_advice[game].append(current_player1_advice)
                            except Exception as e:
                                logger.error(f"Error generating advice: {str(e)}")
                                analysis_prompt_error = {
                                    "error": str(e),
                                    "prompt": advice_prompt
                                }
                                current_player0_advice = f"Error: {str(e)}"
                                current_player1_advice = f"Error: {str(e)}"
                                player0_advice[game].append(current_player0_advice)
                                player1_advice[game].append(current_player1_advice)
                            finally:
                                # Restore original system prompts
                                agents[0].system_prompt = original_player0_prompt
                                agents[1].system_prompt = original_player1_prompt
                            
                            # 2. SECOND TASK: Get scoring from Player 0 only
                            scoring_prompt = (
                                f"Score this game on a scale of 0-10.\n"
                                f"Game outcome: {current_game_history['outcome']}\n"
                                f"Final observation: {last_move['observation']}\n"
                                f"Final action: {last_move['action']}\n\n"
                                f"Rating scale:\n"
                                f"0-2: Poor - Basic mistakes, no strategy\n"
                                f"3-4: Fair - Some good moves but inconsistent\n"
                                f"5-6: Good - Solid play with clear strategy\n"
                                f"7-8: Very Good - Strong tactical play\n"
                                f"9-10: Excellent - Masterful play with perfect execution\n"
                                f"End your response with 'Score: X/10' where X is your rating."
                            )
                            
                            try:
                                # Save original system prompt
                                original_player0_prompt = agents[0].system_prompt
                                
                                # Update system prompt for scoring task
                                agents[0].system_prompt = "You are an expert game judge. Your task is to evaluate and score the game quality on a scale of 0-10."
                                
                                # Get score from Player 0
                                try:
                                    current_player0_score = agents[0](scoring_prompt)
                                    if current_player0_score is None or "Error:" in current_player0_score:
                                        logger.error(f"Error getting score from Player 0: {current_player0_score}")
                                        current_player0_score = "Error generating score"
                                        game_scores[game].append(0)
                                    else:
                                        player0_scores[game].append(current_player0_score)
                                        
                                        # Extract score
                                        current_score = extract_score(current_player0_score)
                                        if current_score is not None:
                                            game_scores[game].append(current_score)
                                            logger.info(f"Game {game_num + 1} scored {current_score}/10")
                                        else:
                                            logger.warning(f"Could not extract score from Player0's response: {current_player0_score}")
                                            game_scores[game].append(0)
                                except Exception as e:
                                    logger.error(f"Exception getting score from Player 0: {type(e).__name__}: {str(e)}")
                                    current_player0_score = f"Error: {type(e).__name__}: {str(e)}"
                                    player0_scores[game].append(current_player0_score)
                                    game_scores[game].append(0)
                            except Exception as e:
                                logger.error(f"Error generating score: {str(e)}")
                                scoring_prompt_error = {
                                    "error": str(e),
                                    "prompt": scoring_prompt
                                }
                                current_player0_score = f"Error: {str(e)}"
                                player0_scores[game].append(current_player0_score)
                                game_scores[game].append(0)
                            finally:
                                # Restore original system prompt
                                agents[0].system_prompt = original_player0_prompt
                                
                        except Exception as e:
                            logger.error(f"Error in analysis/scoring process: {str(e)}")
                            current_player0_advice = f"Error: {str(e)}"
                            current_player1_advice = f"Error: {str(e)}"
                            current_player0_score = f"Error: {str(e)}"
                            player0_advice[game].append(current_player0_advice)
                            player1_advice[game].append(current_player1_advice)
                            player0_scores[game].append(current_player0_score)
                            game_scores[game].append(0)
                else:
                    logger.info(f"Past history limit ({history_limit}), skipping analysis generation for game {game_num + 1}")
                    # Still need to add placeholder entries to maintain array indices
                    player0_advice[game].append(None)
                    player1_advice[game].append(None)
                    player0_scores[game].append(None)
                    game_scores[game].append(0)
                
                game_histories[game].append(current_game_history)
                
                # Log results with model names
                with open(output_file, "a") as f:
                    json.dump({
                        "game": game,
                        "game_num": game_num,
                        "rewards": rewards,
                        "history": current_game_history,
                        "player0_advice": current_player0_advice,
                        "player1_advice": current_player1_advice,
                        "player0_score": current_player0_score,
                        "score": current_score,
                        "history_prompt": history_prompt,
                        "game_summary": game_summaries.get(game, ""),
                        "player0_model": "qwen2.5-32b-chat",
                        "player1_model": player1_model_name,
                        "history_prompt_error": prompt_error,
                        "analysis_prompt_error": analysis_prompt_error,
                        "scoring_prompt_error": scoring_prompt_error,
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
                        "player1_advice": None,
                        "player0_score": None,
                        "score": None,
                        "history_prompt": "",
                        "game_summary": game_summaries.get(game, ""),
                        "player0_model": "qwen2.5-32b-chat",
                        "player1_model": player1_model_name,
                        "error": str(e)
                    }, f)
                    f.write("\n")
                continue

def run_single_setting(setting_num: int, game: str, model_name: str, output_file: str, num_rounds: int, history_limit: Optional[int] = None):
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
            setting4_combined_learning(model_name, [game], output_file, num_rounds, history_limit)
        else:
            raise ValueError(f"Invalid setting number: {setting_num}")
        
        logger.info(f"Setting {setting_num} completed for game {game}")
        return True
    except Exception as e:
        logger.error(f"Error in Setting {setting_num} for game {game}: {str(e)}")
        return False

def parse_games_input(games_str: str) -> List[str]:
    """Parse comma-separated games string into a list of games.
    
    Args:
        games_str: Comma-separated string of games
        
    Returns:
        List of game names with whitespace removed
    """
    if not games_str:
        raise ValueError("Games input cannot be empty")
    
    # Split by comma and strip whitespace
    games = [game.strip() for game in games_str.split(',')]
    
    # Remove any empty strings
    games = [game for game in games if game]
    
    if not games:
        raise ValueError("No valid games found in input")
        
    return games

def parse_settings_input(settings_str: str) -> List[int]:
    """Parse comma-separated settings string into a list of integers.
    
    Args:
        settings_str: Comma-separated string of settings (e.g. '1,2,3')
        
    Returns:
        List of setting numbers
    """
    if not settings_str:
        raise ValueError("Settings input cannot be empty")
    
    try:
        # Split by comma and convert to integers
        settings = [int(setting.strip()) for setting in settings_str.split(',')]
        
        # Validate settings
        valid_settings = {1, 2, 3, 4}
        invalid_settings = [s for s in settings if s not in valid_settings]
        if invalid_settings:
            raise ValueError(f"Invalid settings found: {invalid_settings}. Valid settings are: {sorted(valid_settings)}")
        
        return settings
    except ValueError as e:
        raise ValueError(f"Error parsing settings: {str(e)}")

def run_parallel_evaluation(model_name: str, output_dir: str, games: List[str], settings: List[int], max_concurrent: int = 9, num_rounds: int = 10, history_limit: Optional[int] = None):
    """Run all games and settings in parallel with a maximum of concurrent tasks"""
    logger.info(f"Starting parallel evaluation with max {max_concurrent} concurrent tasks")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create task queue with all combinations
    task_queue = Queue()
    for game in games:
        for setting in settings:
            output_file = os.path.join(output_dir, f"setting{setting}_{game}_{model_name}_results.jsonl")
            task_queue.put((setting, game, output_file, num_rounds, history_limit))
    
    # Track completed and failed tasks
    completed_tasks = set()
    failed_tasks = []
    task_lock = Lock()
    
    def worker():
        while True:
            try:
                # Get next task
                setting, game, output_file, num_rounds, history_limit = task_queue.get_nowait()
                task_id = f"{setting}_{game}"
                
                # Run the task
                success = run_single_setting(setting, game, model_name, output_file, num_rounds, history_limit)
                
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
            success = run_single_setting(setting, game, model_name, output_file, num_rounds, history_limit)
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
    parser.add_argument("--games", type=str, required=True, help="Comma-separated list of games to evaluate (e.g. 'TicTacToe-v0,Poker-v0')")
    parser.add_argument("--settings", type=str, default="1,2,3", help="Comma-separated list of settings to run (e.g. '1,2,3' or '2,3' or '1')")
    parser.add_argument("--history-limit", type=int, help="Limit the number of rounds to keep history for (only applies to setting 4)")
    args = parser.parse_args()
    
    logger.info(f"Starting experiment with arguments: {args}")
    
    # Parse games input
    try:
        selected_games = parse_games_input(args.games)
        logger.info(f"Parsed games: {selected_games}")
    except ValueError as e:
        logger.error(f"Error parsing games input: {str(e)}")
        sys.exit(1)
    
    # Parse settings input
    try:
        selected_settings = parse_settings_input(args.settings)
        logger.info(f"Parsed settings: {selected_settings}")
    except ValueError as e:
        logger.error(f"Error parsing settings input: {str(e)}")
        sys.exit(1)
    
    logger.info(f"Starting vLLM server for {args.model_name}...")
    server_proc = start_vllm_server(args.model_path, args.model_name, port=args.port, gpu=args.gpu)
    
    try:
        run_parallel_evaluation(args.model_name, args.output_dir, selected_games, selected_settings, args.max_concurrent, args.num_rounds, args.history_limit)
        logger.info("All evaluations completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
    finally:
        stop_vllm_server(server_proc)

if __name__ == "__main__":
    main() 