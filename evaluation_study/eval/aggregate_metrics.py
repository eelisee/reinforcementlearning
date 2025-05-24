# eval/aggregate_metrics.py

import os
import pandas as pd

def aggregate_reward_curves(base_path):
    """Aggregiert alle reward_curve.csv-Dateien in einem Ordnerbaum."""
    records = []

    for root, _, files in os.walk(base_path):
        for file in files:
            if file == "reward_curve.csv":
                path = os.path.join(root, file)

                parts = os.path.basename(root).split("_")
                if len(parts) < 3:
                    continue

                seed = parts[-1]
                algo = parts[-2]
                env = "_".join(parts[:-2])

                df = pd.read_csv(path)
                df["algorithm"] = algo
                df["seed"] = int(seed)
                df["env"] = env

                records.append(df)

    if not records:
        raise RuntimeError("Keine reward_curve.csv-Dateien gefunden.")

    return pd.concat(records, ignore_index=True)