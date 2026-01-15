"""eda.py
Tách từ file gốc `books_recommend_f.py`.

Chức năng:
- Thống kê mô tả rating
- Vẽ các biểu đồ EDA và lưu `eda_analysis.png`
"""

from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt


def run_eda(filtered_ratings: pd.DataFrame, save_path: str = "eda_analysis.png") -> None:
    print("\n Rating Distribution:")
    print(filtered_ratings["Book-Rating"].describe())

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Rating distribution
    axes[0, 0].hist(filtered_ratings["Book-Rating"], bins=10,
                    edgecolor='black', alpha=0.7)
    axes[0, 0].set_title("Distribution of Ratings")
    axes[0, 0].set_xlabel("Rating")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].grid(alpha=0.3)

    # 2. Ratings per user
    user_rating_counts = filtered_ratings.groupby("User-ID").size()
    axes[0, 1].hist(user_rating_counts, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].set_title("Ratings per User (After Filtering)")
    axes[0, 1].set_xlabel("Number of Ratings")
    axes[0, 1].set_ylabel("Number of Users")
    axes[0, 1].grid(alpha=0.3)

    # 3. Ratings per book
    book_rating_counts = filtered_ratings.groupby("ISBN").size()
    axes[1, 0].hist(book_rating_counts, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].set_title("Ratings per Book (After Filtering)")
    axes[1, 0].set_xlabel("Number of Ratings")
    axes[1, 0].set_ylabel("Number of Books")
    axes[1, 0].grid(alpha=0.3)

    # 4. Average rating distribution
    avg_ratings = filtered_ratings.groupby("ISBN")["Book-Rating"].mean()
    axes[1, 1].hist(avg_ratings, bins=20, edgecolor='black', alpha=0.7)
    axes[1, 1].set_title("Distribution of Average Book Ratings")
    axes[1, 1].set_xlabel("Average Rating")
    axes[1, 1].set_ylabel("Number of Books")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
