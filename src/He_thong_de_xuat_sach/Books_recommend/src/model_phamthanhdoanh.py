"""model_hovaten.py
Tách từ file gốc `books_recommend_f.py`.

Bao gồm:
- Baseline popularity-based
- Matrix Factorization (PyTorch)
- Train epoch + evaluate (RMSE/MAE trả về)
- Recommend top-N (không gợi ý sách đã đọc)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def popularity_baseline(filtered_ratings: pd.DataFrame, books: pd.DataFrame, min_ratings: int = 20, top_n: int = 20) -> pd.DataFrame:
    popularity_df = (filtered_ratings.groupby("ISBN")
                     .agg({"Book-Rating": ["count", "mean"]})
                     .reset_index())
    popularity_df.columns = ["ISBN", "rating_count", "avg_rating"]

    popularity_df = popularity_df[popularity_df["rating_count"] >= min_ratings].copy()
    popularity_df["popularity_score"] = popularity_df["avg_rating"] * np.log1p(popularity_df["rating_count"])
    popularity_df = popularity_df.sort_values("popularity_score", ascending=False)

    top = popularity_df.head(top_n).merge(
        books[["ISBN", "Book-Title", "Book-Author", "Year-Of-Publication"]],
        on="ISBN", how="left"
    )
    return top


def user_based_train_test_split(data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    np.random.seed(random_state)
    train_list, test_list = [], []

    for user_id in data["User-ID"].unique():
        user_data = data[data["User-ID"] == user_id]
        user_data_shuffled = user_data.sample(frac=1, random_state=random_state)
        n_test = max(1, int(len(user_data) * test_size))
        test_list.append(user_data_shuffled.iloc[:n_test])
        train_list.append(user_data_shuffled.iloc[n_test:])

    train_df = pd.concat(train_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)
    return train_df, test_df


class RatingsDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.users = torch.LongTensor(data["user_idx"].values)
        self.items = torch.LongTensor(data["item_idx"].values)
        self.ratings = torch.FloatTensor(data["Book-Rating"].values)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


class MatrixFactorization(nn.Module):
    def __init__(self, n_users: int, n_items: int, n_factors: int = 100):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        self.user_biases = nn.Embedding(n_users, 1)
        self.item_biases = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.user_factors.weight, std=0.01)
        nn.init.normal_(self.item_factors.weight, std=0.01)
        nn.init.zeros_(self.user_biases.weight)
        nn.init.zeros_(self.item_biases.weight)

    def forward(self, user, item):
        user_embedding = self.user_factors(user)
        item_embedding = self.item_factors(item)
        dot = (user_embedding * item_embedding).sum(dim=1, keepdim=True)
        prediction = dot + self.user_biases(user) + self.item_biases(item) + self.global_bias
        return prediction.squeeze()


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for users, items, ratings in loader:
        users, items, ratings = users.to(device), items.to(device), ratings.to(device)
        predictions = model(users, items)
        loss = criterion(predictions, ratings)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(ratings)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_rmse_mae(model, loader, device):
    model.eval()
    preds, trues = [], []
    for users, items, ratings in loader:
        users, items, ratings = users.to(device), items.to(device), ratings.to(device)
        predictions = model(users, items)
        preds.append(predictions.detach().cpu().numpy())
        trues.append(ratings.detach().cpu().numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    preds = np.clip(preds, 1, 10)

    rmse = np.sqrt(mean_squared_error(trues, preds))
    mae = mean_absolute_error(trues, preds)
    return float(rmse), float(mae)


def train_mf(train_data: pd.DataFrame, test_data: pd.DataFrame, n_users: int, n_items: int,
             batch_size: int = 1024, n_factors: int = 100, lr: float = 0.001, weight_decay: float = 1e-5,
             epochs: int = 20):
    device = get_device()
    model = MatrixFactorization(n_users, n_items, n_factors).to(device)

    train_loader = DataLoader(RatingsDataset(train_data), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(RatingsDataset(test_data), batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses, test_rmses, test_maes = [], [], []
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        test_rmse, test_mae = evaluate_rmse_mae(model, test_loader, device)
        train_losses.append(train_loss)
        test_rmses.append(test_rmse)
        test_maes.append(test_mae)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}/{epochs} | Train Loss: {train_loss:.4f} | Test RMSE: {test_rmse:.4f} | Test MAE: {test_mae:.4f}")

    return model, device, train_losses, test_rmses, test_maes


def recommend_books(user_id: int, model: nn.Module, device: torch.device, filtered_ratings: pd.DataFrame,
                    books: pd.DataFrame, user_id_map: dict, idx_to_isbn: dict, n_items: int, n: int = 10):
    if user_id not in user_id_map:
        raise ValueError(f"User {user_id} not found!")

    user_idx = user_id_map[user_id]
    user_rated_books = set(filtered_ratings[filtered_ratings["User-ID"] == user_id]["ISBN"])

    candidate_indices = [idx for idx in range(n_items) if idx_to_isbn[idx] not in user_rated_books]
    model.eval()
    with torch.no_grad():
        users_tensor = torch.LongTensor([user_idx] * len(candidate_indices)).to(device)
        items_tensor = torch.LongTensor(candidate_indices).to(device)
        predictions = model(users_tensor, items_tensor).cpu().numpy()

    top_n_indices = np.argsort(predictions)[::-1][:n]
    top_item_indices = [candidate_indices[i] for i in top_n_indices]
    top_preds = [float(predictions[i]) for i in top_n_indices]
    top_isbns = [idx_to_isbn[i] for i in top_item_indices]

    result_df = pd.DataFrame({"ISBN": top_isbns, "Predicted_Rating": top_preds})
    result_df = result_df.merge(
        books[["ISBN", "Book-Title", "Book-Author", "Year-Of-Publication", "Publisher"]],
        on="ISBN", how="left"
    )
    return result_df
