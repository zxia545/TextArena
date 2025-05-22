#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import json
import csv
import re
import argparse
from collections import defaultdict
import sys
from typing import Dict, List, Tuple, Set, Optional, Any
from transformers import GPT2Tokenizer
import pandas as pd

# Initialize the tokenizer globally
try:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
except Exception as e:
    print(f"Warning: Failed to load GPT-2 tokenizer: {e}")
    print("Will fall back to approximation method")
    tokenizer = None

def count_tokens(text: str) -> int:
    """Count tokens using GPT-2 tokenizer"""
    if not text:
        return 0
    
    if tokenizer:
        # Use the actual GPT-2 tokenizer
        tokens = tokenizer.encode(text)
        return len(tokens)
    else:
        # Fall back to approximation if tokenizer not available
        return len(text.split()) * 4 // 3  # ~1.33 tokens per word on average

def count_words(text: str) -> int:
    """Count words in text"""
    if not text:
        return 0
    return len(text.split())

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Process game results and generate statistics")
    parser.add_argument("--base_dir", type=str, default ="./results_qwen2.5_32b_0520_setting4_t1_v5", help="Base directory containing model subdirectories")
    parser.add_argument("--type2", type=bool, default=False, help="Process type2 data where error, score, and player0_advice are null")
    return parser.parse_args()

def main():
    args = parse_args()
    
    BASE_DIR = args.base_dir
    is_type2 = args.type2
    
    # Auto-detect model directories
    MODEL_DIRS = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
    if not MODEL_DIRS:
        print(f"No model directories found in {BASE_DIR}")
        sys.exit(1)
    
    # Collect all settings
    settings_set = set()
    file_regex = re.compile(r"(setting\d+)_(.+?)_")
    
    # Stats dictionaries
    stats_all = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    stats_by_setting = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    
    # Token and word count statistics
    token_stats = {}  # {(model, game, setting): [total_tokens, count]}
    word_stats = {}   # {(model, game, setting): [total_words, count]}
    
    # Process each model directory
    for model in MODEL_DIRS:
        model_path = os.path.join(BASE_DIR, model)
        if not os.path.isdir(model_path):
            print(f"⚠️ Directory does not exist: {model_path}, skipping this model")
            continue
        
        for jsonl_path in glob.glob(os.path.join(model_path, "*.jsonl")):
            m = file_regex.search(os.path.basename(jsonl_path))
            if not m:
                print(f"❓ Cannot parse setting/game: {jsonl_path}")
                continue
                
            setting, game = m.groups()
            settings_set.add(setting)
            
            valid_game_count = 0
            
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"⚠️ JSON parse error {jsonl_path}:{lineno}")
                        continue
                    
                    error = obj.get("error")
                    
                    # Handle type2 data differently
                    if is_type2 and setting == "setting4":
                        # Only count when error, score, and player0_advice are all null
                        if error is not None or obj.get("score") is not None or obj.get("player0_advice") is not None:
                            continue
                    elif error is not None:
                        print(f"⚠️ History result error: {jsonl_path}:{lineno}")
                        continue
                    
                    rewards = obj.get("rewards")
                    if not isinstance(rewards, dict):
                        continue
                        
                    try:
                        r0 = float(rewards.get("0"))
                        r1 = float(rewards.get("1"))
                    except (TypeError, ValueError):
                        continue
                    
                    valid_game_count += 1
                    stats_all[model][game][1] += 1
                    stats_by_setting[(model, setting)][game][1] += 1
                    
                    if r1 > r0:
                        stats_all[model][game][0] += 1
                        stats_by_setting[(model, setting)][game][0] += 1
                    
                    # Process history for token and word counts if available
                    # Only look at history["move"][-1]["observation"]
                    history = obj.get("history")
                    if history and isinstance(history, dict) and "moves" in history and history["moves"]:
                        try:
                            moves = history["moves"]
                            if moves and isinstance(moves, list) and len(moves) > 0:
                                last_move = moves[-1]
                                if isinstance(last_move, dict) and "observation" in last_move:
                                    observation = last_move.get("observation", "")
                                    if observation:
                                        try:
                                            tokens = count_tokens(observation)
                                            words = count_words(observation)
                                            
                                            # Initialize if not already initialized
                                            if (model, game, setting) not in token_stats:
                                                token_stats[(model, game, setting)] = [0, 0]
                                            if (model, game, setting) not in word_stats:
                                                word_stats[(model, game, setting)] = [0, 0]
                                                
                                            # Update stats
                                            token_stats[(model, game, setting)][0] += tokens
                                            token_stats[(model, game, setting)][1] += 1
                                            
                                            word_stats[(model, game, setting)][0] += words
                                            word_stats[(model, game, setting)][1] += 1
                                        except Exception as e:
                                            print(f"⚠️ Error processing tokens in {jsonl_path}:{lineno} - {e}")
                                            print(f"   Observation type: {type(observation)}, Tokens: {tokens if 'tokens' in locals() else 'N/A'}")
                        except Exception as e:
                            print(f"⚠️ Error processing history in {jsonl_path}:{lineno} - {e}")
                            print(f"   History structure: {type(history)}, Moves type: {type(history.get('moves', None)) if isinstance(history, dict) else 'N/A'}")
    
    # Convert settings_set to a sorted list
    SETTINGS = sorted(list(settings_set))
    
    # Print results
    def print_block(title, block_stats, key_fmt):
        print(f"\n=== {title} ===")
        for key in sorted(block_stats):
            print(key_fmt(key))
            for game, (win, total) in sorted(block_stats[key].items()):
                acc = win / total if total else 0
                print(f"  {game:<25} {win:>4}/{total:<4}  acc={acc:.3f}")
            print()
    
    print_block("Overall (all settings combined)", stats_all, lambda k: f"\n{str(k)}")
    print_block("By Setting", stats_by_setting, lambda k: f"\n{str(k[0])}  {k[1]}")
    
    # Calculate average token and word counts
    avg_tokens_dict = {}
    avg_words_dict = {}
    
    for (model, game, setting), (total_tokens, count) in token_stats.items():
        avg_tokens_dict[(model, game, setting)] = total_tokens / count if count else 0
        
    for (model, game, setting), (total_words, count) in word_stats.items():
        avg_words_dict[(model, game, setting)] = total_words / count if count else 0
    
    # Write Excel files
    def write_excel(path, header, rows):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = pd.DataFrame(rows, columns=header)
        df.to_excel(path, index=False)
        print(f"✅ Written to {path}")
    
    # Write vertical Excel files
    rows_all = []
    for model, gdict in stats_all.items():
        for game, (win, total) in gdict.items():
            acc = round(win / total, 6) if total else 0
            rows_all.append([model, game, win, total, acc])
    write_excel(os.path.join(BASE_DIR, "acc_all_settings.xlsx"),
              ["model", "game", "win_for_1", "total", "acc"],
              rows_all)
    
    rows_setting = []
    for (model, setting), gdict in stats_by_setting.items():
        for game, (win, total) in gdict.items():
            acc = round(win / total, 6) if total else 0
            
            # Add token and word stats if available
            avg_token = avg_tokens_dict.get((model, game, setting), "")
            avg_word = avg_words_dict.get((model, game, setting), "")
            
            rows_setting.append([model, setting, game, win, total, acc, 
                                 round(avg_token, 2) if avg_token else "", 
                                 round(avg_word, 2) if avg_word else ""])
    
    write_excel(os.path.join(BASE_DIR, "acc_by_setting.xlsx"),
              ["model", "setting", "game", "win_for_1", "total", "acc", "avg_tokens", "avg_words"],
              rows_setting)
    
    # Write horizontal pivot Excel files
    # Collect all game names
    all_games = sorted({game for _, gdict in stats_by_setting.items() for game in gdict})
    
    header_pivot = ["setting", "model"] + all_games
    rows_pivot = []
    
    for setting in SETTINGS:
        for model in MODEL_DIRS:
            row = [setting, model]
            gdict = stats_by_setting.get((model, setting), {})
            for game in all_games:
                win, total = gdict.get(game, (0, 0))
                acc = round(win / total, 6) if total else ""
                row.append(acc)
            rows_pivot.append(row)
    
    write_excel(os.path.join(BASE_DIR, "acc_pivot_by_setting.xlsx"),
              header_pivot,
              rows_pivot)
    
    # Write token and word count statistics
    token_rows = []
    for (model, game, setting), (total, count) in token_stats.items():
        avg = round(total / count, 2) if count else 0
        token_rows.append([model, setting, game, total, count, avg])
    
    write_excel(os.path.join(BASE_DIR, "token_stats.xlsx"),
              ["model", "setting", "game", "total_tokens", "count", "avg_tokens"],
              token_rows)
    
    word_rows = []
    for (model, game, setting), (total, count) in word_stats.items():
        avg = round(total / count, 2) if count else 0
        word_rows.append([model, setting, game, total, count, avg])
    
    write_excel(os.path.join(BASE_DIR, "word_stats.xlsx"),
              ["model", "setting", "game", "total_words", "count", "avg_words"],
              word_rows)
    
    # Write token and word count summary by model
    token_summary_path = os.path.join(BASE_DIR, "token_word_summary.xlsx")
    
    # Token summary by model
    model_token_summary = {}
    model_word_summary = {}
    
    for (model, game, setting), (total_tokens, count) in token_stats.items():
        if model not in model_token_summary:
            model_token_summary[model] = [0, 0]
        model_token_summary[model][0] += total_tokens
        model_token_summary[model][1] += count
    
    for (model, game, setting), (total_words, count) in word_stats.items():
        if model not in model_word_summary:
            model_word_summary[model] = [0, 0]
        model_word_summary[model][0] += total_words
        model_word_summary[model][1] += count
    
    # Create summary dataframes
    summary_rows = []
    for model in MODEL_DIRS:
        total_tokens, token_count = model_token_summary.get(model, [0, 0])
        total_words, word_count = model_word_summary.get(model, [0, 0])
        
        avg_tokens = round(total_tokens / token_count, 2) if token_count else 0
        avg_words = round(total_words / word_count, 2) if word_count else 0
        
        summary_rows.append([model, total_tokens, token_count, avg_tokens, total_words, word_count, avg_words])
    
    # Create summary dataframe
    summary_df = pd.DataFrame(
        summary_rows,
        columns=["model", "total_tokens", "token_samples", "avg_tokens", "total_words", "word_samples", "avg_words"]
    )
    
    # Write token and word summary to Excel with proper formatting
    with pd.ExcelWriter(token_summary_path) as writer:
        summary_df.to_excel(writer, sheet_name="Token & Word Summary", index=False)
        
        # Create per-setting summary sheets
        for setting in SETTINGS:
            setting_rows = []
            for model in MODEL_DIRS:
                model_tokens = 0
                model_token_count = 0
                model_words = 0
                model_word_count = 0
                
                for game in all_games:
                    if (model, game, setting) in token_stats:
                        total, count = token_stats[(model, game, setting)]
                        model_tokens += total
                        model_token_count += count
                    
                    if (model, game, setting) in word_stats:
                        total, count = word_stats[(model, game, setting)]
                        model_words += total
                        model_word_count += count
                
                avg_tokens = round(model_tokens / model_token_count, 2) if model_token_count else 0
                avg_words = round(model_words / model_word_count, 2) if model_word_count else 0
                
                setting_rows.append([model, model_tokens, model_token_count, avg_tokens, model_words, model_word_count, avg_words])
            
            setting_df = pd.DataFrame(
                setting_rows,
                columns=["model", "total_tokens", "token_samples", "avg_tokens", "total_words", "word_samples", "avg_words"]
            )
            setting_df.to_excel(writer, sheet_name=f"{setting} Summary", index=False)
    
    print(f"✅ Written token and word summary to {token_summary_path}")
    
    # Write stats for each setting
    for setting in SETTINGS:
        setting_path = os.path.join(BASE_DIR, f"{setting}_stats.xlsx")
        
        # Create a list of dataframes to write to different sheets
        dfs = []
        
        # Total Games table
        total_games_rows = []
        for model in MODEL_DIRS:
            row = [model]
            gdict = stats_by_setting.get((model, setting), {})
            for game in all_games:
                _, total = gdict.get(game, (0, 0))
                row.append(total)
            total_games_rows.append(row)
        
        total_games_df = pd.DataFrame(total_games_rows, columns=["model"] + all_games)
        
        # Total Wins table
        total_wins_rows = []
        for model in MODEL_DIRS:
            row = [model]
            gdict = stats_by_setting.get((model, setting), {})
            for game in all_games:
                win, _ = gdict.get(game, (0, 0))
                row.append(win)
            total_wins_rows.append(row)
        
        total_wins_df = pd.DataFrame(total_wins_rows, columns=["model"] + all_games)
        
        # Accuracy table
        accuracy_rows = []
        for model in MODEL_DIRS:
            row = [model]
            gdict = stats_by_setting.get((model, setting), {})
            for game in all_games:
                win, total = gdict.get(game, (0, 0))
                acc = round(win / total, 4) if total else 0
                row.append(acc)
            accuracy_rows.append(row)
        
        accuracy_df = pd.DataFrame(accuracy_rows, columns=["model"] + all_games)
        
        # Token Count table
        token_count_rows = []
        for model in MODEL_DIRS:
            row = [model]
            for game in all_games:
                key = (model, game, setting)
                avg = avg_tokens_dict.get(key, 0)
                row.append(round(avg, 2) if avg else 0)
            token_count_rows.append(row)
        
        token_count_df = pd.DataFrame(token_count_rows, columns=["model"] + all_games)
        
        # Word Count table
        word_count_rows = []
        for model in MODEL_DIRS:
            row = [model]
            for game in all_games:
                key = (model, game, setting)
                avg = avg_words_dict.get(key, 0)
                row.append(round(avg, 2) if avg else 0)
            word_count_rows.append(row)
        
        word_count_df = pd.DataFrame(word_count_rows, columns=["model"] + all_games)
        
        # Write all dataframes to a single Excel file with multiple sheets
        with pd.ExcelWriter(setting_path) as writer:
            total_games_df.to_excel(writer, sheet_name=f"Total Games ({setting})", index=False)
            total_wins_df.to_excel(writer, sheet_name=f"Total Wins ({setting})", index=False)
            accuracy_df.to_excel(writer, sheet_name=f"Accuracy ({setting})", index=False)
            token_count_df.to_excel(writer, sheet_name=f"Avg Tokens ({setting})", index=False)
            word_count_df.to_excel(writer, sheet_name=f"Avg Words ({setting})", index=False)
        
        print(f"✅ Written {setting} statistics table {setting}_stats.xlsx")

    # Write pivot with sections
    pivot_section_rows = []
    pivot_section_path = os.path.join(BASE_DIR, "acc_pivot_with_section.xlsx")
    
    # Create dataframe for pivot with sections
    df_list = []
    for setting in SETTINGS:
        # Add a section header
        section_df = pd.DataFrame([[setting, ""] + [""] * len(all_games)], columns=["setting", "model"] + all_games)
        
        # Add model data for this setting
        section_data = []
        for model in MODEL_DIRS:
            row = ["", model]
            gdict = stats_by_setting.get((model, setting), {})
            for game in all_games:
                win, total = gdict.get(game, (0, 0))
                acc = round(win / total, 6) if total else ""
                row.append(acc)
            section_data.append(row)
            
        model_df = pd.DataFrame(section_data, columns=["setting", "model"] + all_games)
        df_list.append(pd.concat([section_df, model_df]))
    
    combined_df = pd.concat(df_list)
    combined_df.to_excel(pivot_section_path, index=False)
    print(f"✅ Written section format table acc_pivot_with_section.xlsx")

if __name__ == "__main__":
    main()
