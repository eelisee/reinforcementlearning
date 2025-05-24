# main.py

import argparse
from experiments.run_all import run_all_experiments

def main():
    parser = argparse.ArgumentParser(description="RL Evaluation Study")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate"], default="train")
    args = parser.parse_args()

    if args.mode == "train":
        run_all_experiments()
    elif args.mode == "evaluate":
        from eval.evaluate import evaluate_all
        evaluate_all()

if __name__ == "__main__":
    main()