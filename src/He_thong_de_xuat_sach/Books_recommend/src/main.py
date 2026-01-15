"""main.py
Chạy toàn bộ pipeline dựa trên code gốc `books_recommend_f.py`.

Chạy:
python src/main.py

Trên Kaggle: giữ nguyên BOOKS_PATH/RATINGS_PATH mặc định.
"""

from __future__ import annotations
import os
import pandas as pd
import torch

from preprocessing import (
    load_data,
    split_single_column,
    clean_text_columns,
    convert_types,
    drop_missing,
    filter_explicit,
    sparsity_filter,
    encode_mappings,
)
from eda import run_eda
from model_hovaten import (
    popularity_baseline,
    user_based_train_test_split,
    train_mf,
    recommend_books,
)
from evaluation import calculate_precision_recall_at_k


def main():
    # Default Kaggle paths
    BOOKS_PATH = os.environ.get("BOOKS_PATH", "/kaggle/input/books-dataset/books_data/books.csv")
    RATINGS_PATH = os.environ.get("RATINGS_PATH", "/kaggle/input/books-dataset/books_data/ratings.csv")

    # 1) Load
    books = load_data(BOOKS_PATH)
    ratings = load_data(RATINGS_PATH)
    books = split_single_column(books, "books")
    ratings = split_single_column(ratings, "ratings")

    # 2) Clean
    books = clean_text_columns(books)
    ratings = clean_text_columns(ratings)
    books, ratings = convert_types(books, ratings)
    books, ratings = drop_missing(books, ratings)

    # 3) Explicit + sparsity
    ratings_explicit = filter_explicit(ratings)
    filtered_ratings = sparsity_filter(ratings_explicit, min_user_ratings=10, min_item_ratings=10)

    n_users = filtered_ratings["User-ID"].nunique()
    n_items = filtered_ratings["ISBN"].nunique()
    print(f"Filtered: users={n_users:,} items={n_items:,} ratings={len(filtered_ratings):,}")

    # 4) Encode
    filtered_ratings, mappings = encode_mappings(filtered_ratings)
    idx_to_isbn = mappings.idx_to_isbn
    user_id_map = mappings.user_id_map

    # 5) EDA
    run_eda(filtered_ratings, save_path="eda_analysis.png")

    # 6) Baseline
    top_20_popular = popularity_baseline(filtered_ratings, books, min_ratings=20, top_n=20)
    top_20_popular.to_csv("baseline_popular_books.csv", index=False)

    # 7) Split train/test (user-based)
    train_data, test_data = user_based_train_test_split(
        filtered_ratings[["User-ID", "ISBN", "Book-Rating", "user_idx", "item_idx"]],
        test_size=0.2,
        random_state=42
    )

    # 8) Train MF
    model, device, train_losses, test_rmses, test_maes = train_mf(
        train_data=train_data,
        test_data=test_data,
        n_users=n_users,
        n_items=n_items,
        batch_size=1024,
        n_factors=100,
        lr=0.001,
        weight_decay=1e-5,
        epochs=20
    )

    # 9) Top-K evaluation
    p_at_5, r_at_5 = calculate_precision_recall_at_k(model, test_data, device=device, k=5, threshold=7)
    p_at_10, r_at_10 = calculate_precision_recall_at_k(model, test_data, device=device, k=10, threshold=7)

    print("\nTop-K Metrics (threshold=7):")
    print(f"Precision@5:  {p_at_5:.4f} | Recall@5:  {r_at_5:.4f}")
    print(f"Precision@10: {p_at_10:.4f} | Recall@10: {r_at_10:.4f}")

    # 10) Demo recommend
    sample_users = filtered_ratings["User-ID"].value_counts().head(10).index
    test_user_id = int(sample_users[0])

    recs = recommend_books(
        user_id=test_user_id,
        model=model,
        device=device,
        filtered_ratings=filtered_ratings,
        books=books,
        user_id_map=user_id_map,
        idx_to_isbn=idx_to_isbn,
        n_items=n_items,
        n=10
    )

    recs.to_csv("sample_recommendations.csv", index=False)

    # 11) Save model + metrics
    torch.save(model.state_dict(), "matrix_factorization_model.pth")

    metrics_df = pd.DataFrame({
        "Metric": ["RMSE", "MAE", "Precision@5", "Recall@5", "Precision@10", "Recall@10"],
        "Value": [test_rmses[-1], test_maes[-1], p_at_5, r_at_5, p_at_10, r_at_10]
    })
    metrics_df.to_csv("evaluation_metrics.csv", index=False)
    print("\nSaved: model + metrics + recommendations + plots")


if __name__ == "__main__":
    main()
