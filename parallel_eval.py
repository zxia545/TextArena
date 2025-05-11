import subprocess
import os
import time
import argparse
import logging
from typing import List
import json
from this_utils import start_vllm_server, stop_vllm_server

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename='parallel_eval.log')
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

def run_single_eval(setting_num: int, game: str, model_name: str, output_dir: str):
    """Run a single evaluation task for a specific setting and game"""
    # Create unique output file for this task
    output_file = f"{output_dir}/setting{setting_num}_{game}_{model_name}_results.jsonl"
    
    # Create the command to run the evaluation
    cmd = [
        "python", "experiment_settings.py",
        "--model-name", model_name,
        "--games", game,
        "--setting", str(setting_num),
        "--output", output_file
    ]
    
    # Run the command
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    return process

def run_parallel_evaluation(model_path: str, model_name: str, port: int = 8010, gpu: int = 4):
    """Run parallel evaluation for all settings and games"""
    logger.info(f"Starting parallel evaluation for model: {model_name}")
    
    # Create output directory
    output_dir = f"results_qwen2.5_32b/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Start vLLM server
    logger.info(f"Starting vLLM server for {model_name}...")
    server_proc = start_vllm_server(model_path, model_name, port=port, gpu=gpu)
    
    try:
        # Wait for server to start
        logger.info("Waiting for server to start...")
        # time.sleep(30)
        
        # Create queue of all combinations
        task_queue = []
        for game in SELECTED_GAMES:
            for setting in range(1, 4):  # Settings 1-3
                task_queue.append((game, setting))
        
        # List to store all running processes
        processes = []
        max_concurrent = 9  # Maximum 9 tasks running at once
        
        # Keep running tasks until queue is empty
        while task_queue or processes:
            # Start new tasks if under limit
            while len(processes) < max_concurrent and task_queue:
                game, setting = task_queue.pop(0)  # Get next task
                logger.info(f"Starting evaluation for Setting {setting}, Game: {game}")
                process = run_single_eval(setting, game, model_name, output_dir)
                processes.append((process, setting, game))
                time.sleep(10)
            
            # Check for completed processes
            for i, (process, setting, game) in enumerate(processes):
                if process.poll() is not None:  # Process has finished
                    stdout, stderr = process.communicate()
                    if process.returncode == 0:
                        logger.info(f"Successfully completed Setting {setting}, Game: {game}")
                    else:
                        logger.error(f"Error in Setting {setting}, Game: {game}: {stderr}")
                    processes.pop(i)
                    break
            
            time.sleep(1)  # Prevent CPU overuse
        
        logger.info("All evaluations completed")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
    finally:
        # Stop the server
        logger.info("Stopping vLLM server...")
        stop_vllm_server(server_proc)
        logger.info("vLLM server stopped successfully")

def main():
    parser = argparse.ArgumentParser(description="Run parallel evaluation of multiple settings and games")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    parser.add_argument("--model-name", type=str, required=True, help="Name to serve the model as")
    parser.add_argument("--port", type=int, default=8010, help="Port for the vLLM server")
    parser.add_argument("--gpu", type=int, default=4, help="Number of GPUs to use")
    args = parser.parse_args()
    
    run_parallel_evaluation(args.model_path, args.model_name, args.port, args.gpu)

if __name__ == "__main__":
    main() 