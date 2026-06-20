#!/usr/bin/env python3
"""Extract and analyze benchmark results from Parquet files."""

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd


def load_latest_parquet(results_dir: str = "results/tier1") -> tuple[Path, pd.DataFrame]:
    """Load the most recent parquet file from results directory."""
    files = sorted(glob.glob(f"{results_dir}/sweep_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {results_dir}")

    latest = Path(files[-1])
    df = pd.read_parquet(latest)
    return latest, df


def load_specific_parquet(filepath: str) -> pd.DataFrame:
    """Load a specific parquet file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    return pd.read_parquet(path)


def print_summary(df: pd.DataFrame, filepath: Path) -> None:
    """Print summary statistics."""
    print("=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print(f"File: {filepath.name}")
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}\n")

    # Filter successful runs
    ok = df[df['status'] == 'ok']

    if len(ok) == 0:
        print("No successful runs found.")
        return

    print(f"Successful cells: {len(ok)}/{len(df)}\n")

    # Summary statistics
    print("Accuracy Metrics:")
    print(f"  Exact Match:     {ok['exact_match_pct'].mean():6.1f}% ± {ok['exact_match_pct'].std():5.1f}%")
    print(f"  Normalized:      {ok['normalized_match_pct'].mean():6.1f}% ± {ok['normalized_match_pct'].std():5.1f}%")
    print(f"  Judge Score:     {ok['judge_avg_score'].mean():6.2f}")

    print("\nPerformance Metrics:")
    print(f"  Throughput:      {ok['mean_tps'].mean():6.2f} tok/s")
    print(f"  Latency:         {ok['mean_total_s'].mean():6.2f}s")
    print(f"  TTFT:            {ok['mean_ttft_s'].mean():.2f}s" if ok['mean_ttft_s'].notna().any() else "  TTFT:            N/A")

    print("\nHardware Metrics:")
    print(f"  Peak VRAM:       {ok['peak_vram_mb'].mean():7.0f} MB")
    print(f"  Avg VRAM:        {ok['avg_vram_mb'].mean():7.0f} MB")
    print(f"  Peak Power:      {ok['peak_power_w'].mean():6.1f} W")
    print(f"  Avg Power:       {ok['avg_power_w'].mean():6.1f} W")
    print(f"  GPU Util:        {ok['avg_gpu_util_pct'].mean():6.1f}%")
    print(f"  CPU Util:        {ok['avg_cpu_util_pct'].mean():6.1f}%")


def print_detailed_results(df: pd.DataFrame) -> None:
    """Print detailed results for each cell."""
    print("\n" + "=" * 80)
    print("DETAILED RESULTS BY CELL")
    print("=" * 80)

    for idx, row in df.iterrows():
        config = (
            f"quant={row['quantization']}, "
            f"kv={row['kv_cache']}, "
            f"prefix={row['prefix_cache']}, "
            f"tokens={row['max_new_tokens']}"
        )

        print(f"\nCell {idx}: {row['model_id']}")
        print(f"  Backend: {row['backend']}")
        print(f"  Config:  {config}")
        print(f"  Status:  {row['status']}")

        if row['status'] == 'ok':
            print(f"  Samples: {row['n_samples']}")
            print("  Accuracy:")
            print(f"    • Exact:       {row['exact_match_pct']:6.1f}%")
            print(f"    • Normalized:  {row['normalized_match_pct']:6.1f}%")
            print(f"    • Judge:       {row['judge_avg_score']:6.2f}")
            print("  Performance:")
            print(f"    • Throughput:  {row['mean_tps']:6.2f} tok/s")
            print(f"    • Latency:     {row['mean_total_s']:6.2f}s")
            print("  Hardware:")
            print(f"    • Peak VRAM:   {row['peak_vram_mb']:7.0f} MB")
            print(f"    • Avg VRAM:    {row['avg_vram_mb']:7.0f} MB")
            print(f"    • Avg Power:   {row['avg_power_w']:6.1f} W")
        else:
            print(f"  Error: {row['error']}")


def export_csv(df: pd.DataFrame, output_path: str) -> None:
    """Export results to CSV."""
    # Select key columns for CSV export
    cols = [
        'model_id', 'backend', 'quantization', 'kv_cache', 'prefix_cache',
        'max_new_tokens', 'status', 'error', 'n_samples',
        'exact_match_pct', 'normalized_match_pct', 'judge_avg_score',
        'mean_tps', 'mean_total_s', 'peak_vram_mb', 'avg_vram_mb',
        'avg_power_w', 'avg_gpu_util_pct', 'hw_sample_count', 'experiment'
    ]

    available_cols = [c for c in cols if c in df.columns]
    df_export = df[available_cols]

    df_export.to_csv(output_path, index=False)
    print(f"✓ Exported to {output_path}")


def export_predictions(df: pd.DataFrame, output_dir: str) -> None:
    """Export predictions and references to individual files."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in df.iterrows():
        if row['status'] == 'ok':
            cell_dir = out_dir / f"cell_{idx}"
            cell_dir.mkdir(exist_ok=True)

            # Save predictions and references
            results_data = {
                'prompt': row['prompts'],
                'reference': row['references'],
                'prediction': row['predictions'],
            }

            results_df = pd.DataFrame(results_data)
            results_df.to_csv(cell_dir / 'predictions.csv', index=False)

            # Save summary
            summary = {
                'model': row['model_id'],
                'backend': row['backend'],
                'exact_match_pct': row['exact_match_pct'],
                'normalized_match_pct': row['normalized_match_pct'],
                'mean_tps': row['mean_tps'],
                'mean_total_s': row['mean_total_s'],
            }

            summary_df = pd.DataFrame([summary])
            summary_df.to_csv(cell_dir / 'summary.csv', index=False)

            print(f"✓ Exported cell {idx} to {cell_dir}")


def show_predictions_sample(df: pd.DataFrame, cell_idx: int = 0, n_samples: int = 5) -> None:
    """Show sample predictions from a specific cell."""
    if cell_idx >= len(df):
        print(f"Cell {cell_idx} not found (only {len(df)} cells available)")
        return

    row = df.iloc[cell_idx]
    if row['status'] != 'ok':
        print(f"Cell {cell_idx} failed: {row['error']}")
        return

    print(f"\n{'=' * 80}")
    print(f"SAMPLE PREDICTIONS — Cell {cell_idx}")
    print(f"{'=' * 80}\n")

    preds = row['predictions']
    refs = row['references']

    for i in range(min(n_samples, len(preds))):
        pred_display = preds[i][:70] + "..." if len(preds[i]) > 70 else preds[i]
        match = "✓" if preds[i].strip() == refs[i] else "✗"

        print(f"{match} [{i}] Prediction: {pred_display}")
        print(f"      Reference: {refs[i]}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract and analyze LLM benchmark results from Parquet files"
    )

    parser.add_argument(
        '--file', '-f',
        type=str,
        help='Specific parquet file to load (default: latest)',
    )

    parser.add_argument(
        '--dir', '-d',
        type=str,
        default='results/tier1',
        help='Results directory (default: results/tier1)',
    )

    parser.add_argument(
        '--summary', '-s',
        action='store_true',
        help='Print summary statistics',
    )

    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Print detailed results for each cell',
    )

    parser.add_argument(
        '--export-csv',
        type=str,
        metavar='PATH',
        help='Export results to CSV file',
    )

    parser.add_argument(
        '--export-predictions',
        type=str,
        metavar='DIR',
        help='Export predictions and references to directory',
    )

    parser.add_argument(
        '--show-predictions',
        type=int,
        metavar='CELL_IDX',
        help='Show sample predictions from cell (use with -n)',
    )

    parser.add_argument(
        '-n',
        type=int,
        default=5,
        help='Number of sample predictions to show (default: 5)',
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available parquet files',
    )

    args = parser.parse_args()

    # List available files
    if args.list:
        files = sorted(glob.glob(f"{args.dir}/sweep_*.parquet"))
        print(f"Available parquet files in {args.dir}:\n")
        for i, f in enumerate(files[-10:], 1):  # Show last 10
            print(f"  {i}. {Path(f).name}")
        return

    # Load data
    try:
        if args.file:
            df = load_specific_parquet(args.file)
            filepath = Path(args.file)
        else:
            filepath, df = load_latest_parquet(args.dir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Print summary (default if no other action specified)
    if not any([args.export_csv, args.export_predictions, args.show_predictions is not None]):
        args.detailed = True
        args.summary = True

    if args.summary:
        print_summary(df, filepath)

    if args.detailed:
        print_detailed_results(df)

    if args.export_csv:
        export_csv(df, args.export_csv)

    if args.export_predictions:
        export_predictions(df, args.export_predictions)

    if args.show_predictions is not None:
        show_predictions_sample(df, args.show_predictions, args.n)


if __name__ == '__main__':
    main()
