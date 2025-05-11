from this_utils import start_vllm_server_thinking, stop_vllm_server


import argparse
import subprocess
import sys
import time
from this_utils import start_vllm_server, stop_vllm_server

def run_evaluation(model_name: str):
    """
    Activate conda environment and run evaluation script for the given model.
    """
    eval_cmd = f"""
    source activate agentboard && \
    python agentboard/eval_main.py \
    --cfg-path=eval_configs/main_results_all_tasks.yaml \
    --tasks=all \
    --model="{model_name}" \
    --log_path=./results/{model_name} \
    --project_name="eval-{model_name}" \
    --baseline_dir=./data/baseline_results
    """
    process = subprocess.Popen(
        ["bash", "-c", eval_cmd],
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    process.communicate()
    return process.returncode

def main():
    parser = argparse.ArgumentParser(description="Start vLLM server and evaluate model.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model (e.g., qwen/Qwen2-7B-Instruct)")
    parser.add_argument("--model-name", type=str, required=True, help="Name to serve the model as (e.g., qwen2.5-7b)")
    parser.add_argument("--port", type=int, default=8010, help="Port for the vLLM server (default: 8000)")
    parser.add_argument("--gpu", type=int, default=4, help="Number of GPUs to use (default: 1)")
    parser.add_argument("--just_host", help="Use CPU instead of GPU")
    args = parser.parse_args()

    # Step 1: Start vLLM server
    server_proc = start_vllm_server_thinking(args.model_path, args.model_name, port=args.port, gpu=args.gpu)

    if args.just_host:
        # stop when user control c
        try:
            print("[INFO] vLLM server started. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("[INFO] Stopping vLLM server...")
            stop_vllm_server(server_proc)
            sys.exit(0)
    else:
        try:
            # Step 2: Run evaluation
            print("[INFO] Starting evaluation script...")
            return_code = run_evaluation(args.model_name)

            if return_code == 0:
                print("[INFO] Evaluation completed successfully.")
            else:
                print("[ERROR] Evaluation script exited with code:", return_code)
        finally:
            # Step 3: Stop the server regardless of evaluation success
            stop_vllm_server(server_proc)
    stop_vllm_server(server_proc)

if __name__ == "__main__":
    main()
