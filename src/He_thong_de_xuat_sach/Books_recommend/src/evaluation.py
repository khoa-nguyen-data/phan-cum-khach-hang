"""evaluation.py
Tách từ file gốc `books_recommend_f.py`.

- Precision@K, Recall@K (threshold-based)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import torch


def calculate_precision_recall_at_k(model, data: pd.DataFrame, device: torch.device, k: int = 10, threshold: float = 7.0):
    model.eval()
    user_metrics = {}

    for user_idx in data["user_idx"].unique():
        user_data = data[data["user_idx"] == user_idx]
        if len(user_data) < k:
            continue

        with torch.no_grad():
            users_tensor = torch.LongTensor(user_data["user_idx"].values).to(device)
            items_tensor = torch.LongTensor(user_data["item_idx"].values).to(device)
            predictions = model(users_tensor, items_tensor).cpu().numpy()

        actual_ratings = user_data["Book-Rating"].values
        sorted_indices = np.argsort(predictions)[::-1]
        top_k_indices = sorted_indices[:k]

        relevant_items = actual_ratings >= threshold
        n_relevant = relevant_items.sum()
        if n_relevant == 0:
            continue

        recommended_relevant = relevant_items[top_k_indices].sum()
        precision = recommended_relevant / k
        recall = recommended_relevant / n_relevant

        user_metrics[user_idx] = {"precision": precision, "recall": recall}

    if len(user_metrics) == 0:
        return 0.0, 0.0

    avg_precision = float(np.mean([m["precision"] for m in user_metrics.values()]))
    avg_recall = float(np.mean([m["recall"] for m in user_metrics.values()]))
    return avg_precision, avg_recall
