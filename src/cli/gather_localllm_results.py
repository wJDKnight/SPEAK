#!/usr/bin/env python3
"""
Script to gather results from local LLM experiments.

This script processes JSONL result files from local LLM models (Llama8, Qwen30, etc.)
and extracts predictions using the extract_json_microenvironment function from src/utils.py.
Results are organized by data_name and saved as CSV files.

Directory structure expected:
    examples/intermediates/local_llm_results/STARmap/{experiment_type}/{model_name}/
    
File naming pattern:
    STARmap_{data_name}_{rep}.jsonl
    
Example paths:
    - Llama8/STARmap_BZ5_test_rep1.jsonl
    - Qwen30/STARmap_BZ9_all_rep2.jsonl
"""

import os
import json
import pandas as pd
from pathlib import Path
from src.utils import extract_json_microenvironment
from src.paths import intermediate_root
from typing import List, Dict
import argparse


DEFAULT_LOCAL_RESULTS = str(intermediate_root() / "local_llm_results")


def process_jsonl_file(file_path: str) -> pd.DataFrame:
    """
    Process a single JSONL file and extract predictions.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        DataFrame with extracted predictions and metadata
    """
    results = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                # Parse JSON line
                data = json.loads(line.strip())
                
                # Extract prediction
                predict = data.get('predict', '')
                
                # Get the content (microenvironment prediction)
                prediction = extract_json_microenvironment(predict)['content']
                
                # Store result with line number as index
                results.append({
                    'cell_id': line_num - 1,  # 0-indexed cell ID
                    'prediction': prediction,
                    # 'raw_predict': predict
                })
                
            except json.JSONDecodeError as e:
                print(f"  Warning: JSON decode error at line {line_num} in {file_path}: {e}")
                results.append({
                    'cell_id': line_num - 1,
                    'prediction': 'Error',
                    # 'raw_predict': ''
                })
            except Exception as e:
                print(f"  Warning: Error at line {line_num} in {file_path}: {e}")
                results.append({
                    'cell_id': line_num - 1,
                    'prediction': 'Error',
                    # 'raw_predict': ''
                })
    
    df = pd.DataFrame(results)
    df.set_index('cell_id', inplace=True)
    
    return df


def gather_results(base_dir: str = DEFAULT_LOCAL_RESULTS,
                   dataset_type: str = "STARmap",
                   models: List[str] = None,
                   data_names: List[str] = None,
                   reps: List[str] = None,
                   experiment_types: List[str] = None,
                   output_dir: str = f"{DEFAULT_LOCAL_RESULTS}/processed") -> Dict[str, pd.DataFrame]:
    """
    Gather all local LLM results and organize by data_name.
    
    Args:
        base_dir: Base directory containing results
        dataset_type: Type of dataset (e.g., 'STARmap', 'LIBD')
        models: List of model names (e.g., ['Llama8', 'Qwen30'])
        data_names: List of data names (e.g., ['BZ5_test', 'BZ9_all', 'BZ14_all'])
        reps: List of replicate names (e.g., ['rep1', 'rep2', 'rep3'])
        experiment_types: List of experiment types (e.g., ['finetune_BZ5', 'zeroshot'])
        output_dir: Directory to save processed results
        
    Returns:
        Dictionary mapping data_name to combined DataFrame
    """
    # Default values
    if models is None:
        models = ['Llama8', 'Qwen30']
    
    if data_names is None:
        data_names = ['BZ5_test', 'BZ9_all', 'BZ14_all']
    
    if reps is None:
        reps = ['rep1', 'rep2', 'rep3']
    
    if experiment_types is None:
        experiment_types = ['finetune_BZ5', 'zeroshot']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store results by data_name
    data_results = {}
    
    # Iterate through all combinations
    for experiment_type in experiment_types:
        for model in models:
            for data_name in data_names:
                print(f"\nProcessing {experiment_type}/{model}/{data_name}...")
                
                # List to store DataFrames for all replicates
                replicate_dfs = []
                
                for rep in reps:
                    # Construct file path
                    file_name = f"{dataset_type}_{data_name}_{rep}.jsonl"
                    file_path = os.path.join(
                        base_dir,
                        dataset_type,
                        experiment_type,
                        model,
                        file_name
                    )
                    
                    # Check if file exists
                    if not os.path.exists(file_path):
                        print(f"  Warning: File not found: {file_path}")
                        continue
                    
                    print(f"  Processing {file_name}...")
                    
                    # Process the file
                    df = process_jsonl_file(file_path)
                    
                    # Add metadata columns
                    df['model'] = model
                    df['experiment_type'] = experiment_type
                    df['data_name'] = data_name
                    df['replicate'] = rep
                    
                    replicate_dfs.append(df)
                
                # Combine all replicates for this model/data_name/experiment_type
                if replicate_dfs:
                    combined_df = pd.concat(replicate_dfs, ignore_index=False)
                    
                    # Create a unique key for this combination
                    key = f"{data_name}_{experiment_type}_{model}"
                    
                    # Store in data_results dictionary
                    if data_name not in data_results:
                        data_results[data_name] = []
                    
                    data_results[data_name].append(combined_df)
                    
                    print(f"  Processed {len(replicate_dfs)} replicates, total {len(combined_df)} predictions")
    
    # Combine all results for each data_name
    final_results = {}
    for data_name, df_list in data_results.items():
        if df_list:
            print(f"\nCombining all results for {data_name}...")
            combined = pd.concat(df_list, ignore_index=False)
            final_results[data_name] = combined
            
            # Save to CSV
            output_file = os.path.join(output_dir, f"{data_name}_all_results.csv")
            combined.to_csv(output_file)
            print(f"  Saved {len(combined)} predictions to {output_file}")
            
            # Print summary statistics
            print(f"  Summary for {data_name}:")
            print(f"    - Total predictions: {len(combined)}")
            print(f"    - Models: {combined['model'].unique().tolist()}")
            print(f"    - Experiment types: {combined['experiment_type'].unique().tolist()}")
            print(f"    - Replicates: {combined['replicate'].unique().tolist()}")
            print(f"    - Prediction distribution:")
            print(combined['prediction'].value_counts().head(10))
    
    return final_results


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description='Gather results from local LLM experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all results with default settings
  python gather_localllm_results.py
  
  # Process specific models only
  python gather_localllm_results.py --models Llama8 Qwen30
  
  # Process specific data names
  python gather_localllm_results.py --data_names BZ5_test BZ9_all
  
  # Specify custom directories
  python -m src.cli.gather_localllm_results --base_dir examples/intermediates/local_llm_results --output_dir examples/intermediates/local_llm_results/processed
        """
    )
    
    parser.add_argument(
        '--base_dir',
        type=str,
        default=DEFAULT_LOCAL_RESULTS,
        help='Base directory containing bundled local-LLM result files'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=f'{DEFAULT_LOCAL_RESULTS}/processed',
        help='Output directory for processed local-LLM result tables'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        default=['Llama8', 'Qwen30'],
        help='List of model names to process (default: Llama8 Qwen30)'
    )

    parser.add_argument(
        '--dataset_type',
        type=str,
        default='STARmap',
        help='Type of dataset (default: STARmap)'
    )
    
    parser.add_argument(
        '--data_names',
        nargs='+',
        default=['BZ5_test', 'BZ9_all', 'BZ14_all'],
        help='List of data names to process (default: BZ5_test BZ9_all BZ14_all)'
    )
    
    parser.add_argument(
        '--reps',
        nargs='+',
        default=['rep1', 'rep2', 'rep3'],
        help='List of replicate names (default: rep1 rep2 rep3)'
    )
    
    parser.add_argument(
        '--experiment_types',
        nargs='+',
        default=['finetune_BZ5', 'zeroshot'],
        help='List of experiment types (default: finetune_BZ5 zeroshot)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Local LLM Results Gathering Script")
    print("=" * 80)
    print(f"Base directory: {args.base_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Models: {args.models}")
    print(f"Data names: {args.data_names}")
    print(f"Replicates: {args.reps}")
    print(f"Experiment types: {args.experiment_types}")
    print("=" * 80)
    
    # Gather results
    results = gather_results(
        base_dir=args.base_dir,
        dataset_type=args.dataset_type,
        models=args.models,
        data_names=args.data_names,
        reps=args.reps,
        experiment_types=args.experiment_types,
        output_dir=args.output_dir
    )
    
    print("\n" + "=" * 80)
    print("Processing complete!")
    print(f"Processed {len(results)} unique data_name groups")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    main()
