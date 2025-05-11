import subprocess
import os
import time
import argparse
import logging
import signal
from typing import List, Tuple
from this_utils import start_vllm_server, stop_vllm_server
import sys
# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='parallel_eval.log'
)
logger = logging.getLogger(__name__)

# Selected games for evaluation
SELECTED_GAMES = [
    "TicTacToe-v0",
    "Poker-v0",
    "Stratego-v0",
    "TruthAndDeception-v0",
    "UltimateTicTacToe-v0",
    "Checkers-v0",
    "Othello-v0",
    "SecretMafia-v0"
]

def run_experiment_settings(game: str, model_name: str, output_dir: str, max_workers: int = 3):
    """Run experiment settings for a single game"""
    logger.info(f"Running experiment settings for game: {game}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Run experiment settings
    cmd = [
        "python", "experiment_settings.py",
        "--model-name", model_name,
        "--game", game,
        "--output-dir", output_dir,
        "--max-workers", str(max_workers)
    ]
    
    try:
        process = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Successfully completed settings for {game}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running settings for {game}: {e.stderr}")
        return False

def run_parallel_evaluation(model_path: str, model_name: str, port: int = 8010, gpu: int = 4):
    """Run parallel evaluation for all games"""
    logger.info(f"Starting parallel evaluation for model: {model_name}")
    
    # Create output directory
    output_dir = os.path.join("./results_qwen2.5_32b", model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Start vLLM server
    logger.info(f"Starting vLLM server for {model_name}...")
    server_proc = start_vllm_server(model_path, model_name, port=port, gpu=gpu)
    
    try:
        # Wait for server to initialize
        logger.info("Waiting for server to initialize...")
        time.sleep(30)
        
        # Run each game sequentially
        for game in SELECTED_GAMES:
            logger.info(f"Starting evaluation for game: {game}")
            
            # Run experiment settings for this game
            success = run_experiment_settings(game, model_name, output_dir)
            
            if success:
                logger.info(f"Successfully completed evaluation for {game}")
            else:
                logger.error(f"Failed to complete evaluation for {game}")
            
            # Wait before next game
            logger.info("Waiting 10 seconds before next game...")
            time.sleep(10)
        
        logger.info("All games completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
    finally:
        # Stop the server
        logger.info("Stopping vLLM server...")
        stop_vllm_server(server_proc)
        logger.info("vLLM server stopped successfully")

def main():
    parser = argparse.ArgumentParser(
        description="Run parallel evaluation of multiple games"
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--gpu", type=int, default=4)
    args = parser.parse_args()
    
    try:
        run_parallel_evaluation(
            args.model_path,
            args.model_name,
            port=args.port,
            gpu=args.gpu
        )
    except Exception as e:
        logger.error(f"Failed to run evaluation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
