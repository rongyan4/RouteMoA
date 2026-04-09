#!/usr/bin/env python3
"""
Script for analyzing MOA layer-usage behavior.
Used for experiments on the relationship between STOP_THRESHOLD and layer count.
"""
import json
import os
import sys
import glob
import argparse
from collections import defaultdict
import statistics

def load_layer_logs(log_dir=None, log_file=None):
    """
    Load layer-usage log files.
    
    Args:
        log_dir: Log directory path (load all log files under this directory)
        log_file: Specific log file path (takes precedence if provided)
    
    Returns:
        List of records
    """
    records = []
    
    # If a specific file is provided, load it first
    if log_file is not None:
        if not os.path.exists(log_file):
            print(f"Error: specified log file does not exist: {log_file}")
            return []
        
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        
        print(f"✓ Loaded from specified file: {log_file}")
        return records
    
    # Otherwise, load all log files from the directory
    if log_dir is None:
        log_dir = '/home/1002/wangjize/wuhan/RouteMoA/emoa/logs'
    
    log_pattern = os.path.join(log_dir, "layer_usage_*.jsonl")
    log_files = glob.glob(log_pattern)
    
    if not log_files:
        print(f"No log files found: {log_pattern}")
        return []
    
    for lf in sorted(log_files):
        with open(lf, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    
    return records

def analyze_by_threshold(records):
    """Analyze round distribution by stop threshold."""
    threshold_groups = defaultdict(list)
    
    for record in records:
        threshold = record['stop_threshold']
        threshold_groups[threshold].append(record)
    
    print("\n" + "=" * 80)
    print("Analysis by stop threshold")
    print("=" * 80)
    
    for threshold in sorted(threshold_groups.keys()):
        group = threshold_groups[threshold]
        rounds = [r['actual_rounds'] for r in group]
        
        print(f"\nThreshold: {threshold}")
        print(f"  Total queries: {len(group)}")
        print(f"  Average rounds: {statistics.mean(rounds):.2f}")
        print(f"  Round distribution:")
        
        # # Count distribution per round
        # round_dist = defaultdict(int)
        # for r in rounds:
        #     round_dist[r] += 1
        
        # for r in sorted(round_dist.keys()):
        #     count = round_dist[r]
        #     percentage = (count / len(group)) * 100
        #     print(f"    {r} rounds: {count} ({percentage:.1f}%)")
        
        # Early-stop statistics
        early_stops = sum(1 for r in group if r['early_stop'])
        print(f"  Early-stop rate: {early_stops}/{len(group)} ({early_stops/len(group)*100:.1f}%)")

def analyze_score_vs_rounds(records):
    """Analyze the relationship between final score and number of rounds."""
    print("\n" + "=" * 80)
    print("Final score vs. rounds")
    print("=" * 80)
    
    round_groups = defaultdict(list)
    for record in records:
        round_groups[record['actual_rounds']].append(record['final_score'])
    
    for rounds in sorted(round_groups.keys()):
        scores = round_groups[rounds]
        print(f"\n{rounds} rounds:")
        print(f"  Sample count: {len(scores)}")
        print(f"  Average score: {statistics.mean(scores):.4f}")
        print(f"  Median score: {statistics.median(scores):.4f}")
        if len(scores) > 1:
            print(f"  Standard deviation: {statistics.stdev(scores):.4f}")
        print(f"  Score range: [{min(scores):.4f}, {max(scores):.4f}]")

def analyze_model_usage(records):
    """Analyze model usage."""
    print("\n" + "=" * 80)
    print("Model usage analysis")
    print("=" * 80)
    
    total_models_per_round = []
    for record in records:
        num_models = record['num_models_per_round']
        total_models_per_round.extend(num_models)
        
        # Count average number of models per round
        # avg_models = statistics.mean(num_models)
        # print(f"\nQuery {record['query_id']}:")
        # print(f"  Rounds: {record['actual_rounds']}")
        # print(f"  Models per round: {num_models}")
        # print(f"  Average models per round: {avg_models:.2f}")
        # print(f"  Total model calls: {sum(num_models)}")
    
    print(f"\nOverall statistics:")
    print(f"  Average models per round: {statistics.mean(total_models_per_round):.2f}")
    print(f"  Median: {statistics.median(total_models_per_round):.2f}")

def generate_summary(records):
    """Generate an overall summary."""
    print("\n" + "=" * 80)
    print("Overall summary")
    print("=" * 80)
    
    total = len(records)
    rounds = [r['actual_rounds'] for r in records]
    scores = [r['final_score'] for r in records]
    early_stops = sum(1 for r in records if r['early_stop'])
    
    print(f"\nTotal queries: {total}")
    print(f"Average rounds: {statistics.mean(rounds):.2f}")
    print(f"Median rounds: {statistics.median(rounds):.2f}")
    print(f"\nAverage final score: {statistics.mean(scores):.4f}")
    print(f"Median score: {statistics.median(scores):.4f}")
    print(f"\nEarly stops: {early_stops}/{total} ({early_stops/total*100:.1f}%)")
    
    # Round distribution
    print(f"\nRound distribution:")
    round_dist = defaultdict(int)
    for r in rounds:
        round_dist[r] += 1
    
    for r in sorted(round_dist.keys()):
        count = round_dist[r]
        percentage = (count / total) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {r} rounds: {count:4d} ({percentage:5.1f}%) {bar}")

def export_for_experiment(records, output_file):
    """Export in experiment data format (CSV)."""
    import csv
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'query_id', 'timestamp', 'stop_threshold', 'actual_rounds',
            'stop_reason', 'final_score', 'early_stop', 'total_model_calls'
        ])
        
        for record in records:
            writer.writerow([
                record['query_id'],
                record['timestamp'],
                record['stop_threshold'],
                record['actual_rounds'],
                record['stop_reason'],
                record['final_score'],
                record['early_stop'],
                sum(record['num_models_per_round'])
            ])
    
    print(f"\n✓ Experiment data exported to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze MOA layer usage')
    parser.add_argument('--log-dir', default='/home/1002/wangjize/wuhan/RouteMoA/emoa/logs',
                       help='Log directory path (mutually exclusive with --log-file)')
    parser.add_argument('--log-file', help='Specify log file path (takes precedence)')
    parser.add_argument('--export-csv', help='Export CSV file path')
    parser.add_argument('--threshold', type=float, help='Filter data by specific threshold')
    
    args = parser.parse_args()
    
    # Check argument conflicts
    if args.log_file is not None and args.log_dir != '/home/1002/wangjize/wuhan/RouteMoA/emoa/logs':
        print("Error: --log-file and --log-dir cannot be used together")
        return
    
    # Load data
    records = load_layer_logs(log_dir=args.log_dir, log_file=args.log_file)
    if not records:
        print("No records found, exiting")
        return
    
    print(f"✓ Loaded {len(records)} records")
    
    # If a threshold is provided, filter records
    if args.threshold is not None:
        records = [r for r in records if r['stop_threshold'] == args.threshold]
        print(f"✓ {len(records)} records remain after filtering (threshold={args.threshold})")
    
    # Run analysis
    generate_summary(records)
    analyze_by_threshold(records)
    analyze_score_vs_rounds(records)
    analyze_model_usage(records)
    
    # Export CSV
    # if args.export_csv:
    #     export_for_experiment(records, args.export_csv)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
