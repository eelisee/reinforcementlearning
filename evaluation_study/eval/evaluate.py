# eval/evaluate.py

import argparse
from aggregate_metrics import aggregate_reward_curves
from plot_metrics import plot_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="evaluation_study/data", help="Pfad zum Ordner mit reward_curve.csv-Dateien")
    parser.add_argument("--output_dir", type=str, default="evaluation_study/plots", help="Wo sollen die Plots gespeichert werden?")
    args = parser.parse_args()

    df = aggregate_reward_curves(args.data_dir)
    plot_metrics(df, args.output_dir)

if __name__ == "__main__":
    main()