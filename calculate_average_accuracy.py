#!/usr/bin/env python3
"""
Calculate average accuracy for EPIC_inst, EPIC_inst_qwen, and EPIC_inst_oss
across all evaluation folders.
"""

import pandas as pd
import os
from pathlib import Path

# Define the folders and their expected persona counts
FOLDERS = {
    'prefwiki': {
        'path': 'output_prefwiki/wiki/evaluation_report.csv',
        'persona_count': 57
    },
    'prefeval': {
        'path': 'output_prefeval/lmsys_sampled/evaluation_report.csv',
        'persona_count': 57
    },
    'prefeli5': {
        'path': 'output_prefeli5/eli5/evaluation_report.csv',
        'persona_count': 73
    },
    'prefrq': {
        'path': 'output_rq/wiki/evaluation_report.csv',
        'persona_count': 90
    }
}

# Methods to analyze
TARGET_METHODS = ['EPIC_inst', 'EPIC_inst_qwen', 'EPIC_inst_oss']

def load_and_filter_data(file_path, methods):
    """Load CSV and filter by methods."""
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    
    # Filter by methods
    filtered_df = df[df['method'].isin(methods)].copy()
    
    return filtered_df

def calculate_average_accuracy(df, method, folder_name):
    """Calculate average accuracy for a specific method."""
    method_df = df[df['method'] == method]
    
    if len(method_df) == 0:
        return None, 0
    
    # Get accuracy column (handle both % and non-% formats)
    accuracy_col = 'preference_following_accuracy(%)'
    if accuracy_col not in method_df.columns:
        # Try alternative column names
        for col in method_df.columns:
            if 'accuracy' in col.lower():
                accuracy_col = col
                break
    
    accuracies = method_df[accuracy_col].astype(float)
    avg_accuracy = accuracies.mean()
    count = len(method_df)
    
    return avg_accuracy, count

def main():
    base_dir = Path(__file__).parent
    
    results = {}
    
    print("=" * 80)
    print("Average Accuracy Calculation")
    print("=" * 80)
    print()
    
    # Process each folder
    for folder_name, folder_info in FOLDERS.items():
        file_path = base_dir / folder_info['path']
        expected_count = folder_info['persona_count']
        
        print(f"Processing {folder_name} (expected {expected_count} personas)...")
        print(f"  File: {file_path}")
        
        df = load_and_filter_data(file_path, TARGET_METHODS)
        
        if df is None:
            print(f"  ⚠️  Skipped (file not found)")
            print()
            continue
        
        folder_results = {}
        
        for method in TARGET_METHODS:
            avg_acc, count = calculate_average_accuracy(df, method, folder_name)
            
            if avg_acc is not None:
                folder_results[method] = {
                    'average_accuracy': avg_acc,
                    'persona_count': count,
                    'expected_count': expected_count
                }
                print(f"  {method:20s}: {avg_acc:6.2f}% ({count}/{expected_count} personas)")
            else:
                folder_results[method] = None
                print(f"  {method:20s}: No data found")
        
        results[folder_name] = folder_results
        print()
    
    # Summary table
    print("=" * 80)
    print("Summary Table")
    print("=" * 80)
    print()
    
    # Header
    header = f"{'Folder':<15} {'Expected':<10} "
    for method in TARGET_METHODS:
        header += f"{method:>20} "
    print(header)
    print("-" * 80)
    
    # Data rows
    for folder_name, folder_info in FOLDERS.items():
        expected_count = folder_info['persona_count']
        row = f"{folder_name:<15} {expected_count:<10} "
        
        if folder_name in results:
            for method in TARGET_METHODS:
                if results[folder_name][method] is not None:
                    avg_acc = results[folder_name][method]['average_accuracy']
                    count = results[folder_name][method]['persona_count']
                    row += f"{avg_acc:6.2f}% ({count:>3}) "
                else:
                    row += f"{'N/A':>20} "
        else:
            row += f"{'N/A':>20} " * len(TARGET_METHODS)
        
        print(row)
    
    print()
    print("=" * 80)
    print("Overall Average (across all folders)")
    print("=" * 80)
    print()
    
    # Calculate overall averages
    for method in TARGET_METHODS:
        accuracies = []
        total_personas = 0
        
        for folder_name in FOLDERS.keys():
            if folder_name in results and results[folder_name][method] is not None:
                method_data = results[folder_name][method]
                # Weight by persona count
                accuracies.append(method_data['average_accuracy'])
                total_personas += method_data['persona_count']
        
        if accuracies:
            overall_avg = sum(accuracies) / len(accuracies)
            weighted_avg = sum(acc * results[folder_name][method]['persona_count'] 
                             for folder_name, folder_info in FOLDERS.items()
                             if folder_name in results and results[folder_name][method] is not None
                             for acc in [results[folder_name][method]['average_accuracy']]) / total_personas if total_personas > 0 else 0
            
            # Simple average (not weighted)
            print(f"{method:20s}: {overall_avg:6.2f}% (simple average)")
            print(f"{'':20s}  {weighted_avg:6.2f}% (weighted by persona count, total: {total_personas})")
        else:
            print(f"{method:20s}: No data available")
        print()

if __name__ == '__main__':
    main()

