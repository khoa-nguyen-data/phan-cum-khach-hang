"""preprocessing.py
Tách từ file gốc `books_recommend_f.py`.

Chức năng:
- Load dữ liệu (hỗ trợ delimiter ';', encoding latin-1)
- Clean dữ liệu, convert kiểu
- Lọc rating > 0 (explicit)
- Lọc sparsity (min ratings per user/item)
- Encode user/item -> index
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV với nhiều cách xử lý delimiter."""
    try:
        return pd.read_csv(file_path, encoding='latin-1', low_memory=False)
    except Exception:
        return pd.read_csv(file_path, sep=';', encoding='latin-1', on_bad_lines='skip', low_memory=False, engine='python')


def split_single_column(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Nếu CSV bị đọc thành 1 cột, split theo ';' để ra đúng cột."""
    if df.shape[1] != 1:
        return df

    col_name = df.columns[0]
    tmp = df[col_name].astype(str).str.replace('"', '').str.strip().str.split(';', expand=True)

    if kind == "books":
        tmp.columns = ["ISBN", "Book-Title", "Book-Author", "Year-Of-Publication",
                       "Publisher", "Image-URL-S", "Image-URL-M", "Image-URL-L"]
    elif kind == "ratings":
        tmp.columns = ["User-ID", "ISBN", "Book-Rating"]
    else:
        raise ValueError("kind must be 'books' or 'ratings'")
    return tmp


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace('"', '').str.strip()
    return df


def convert_types(books: pd.DataFrame, ratings: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ratings["User-ID"] = pd.to_numeric(ratings["User-ID"], errors='coerce')
    ratings["Book-Rating"] = pd.to_numeric(ratings["Book-Rating"], errors='coerce')
    if "Year-Of-Publication" in books.columns:
        books["Year-Of-Publication"] = pd.to_numeric(books["Year-Of-Publication"], errors='coerce')
    return books, ratings


def drop_missing(books: pd.DataFrame, ratings: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    books = books.dropna(subset=["ISBN", "Book-Title"])
    ratings = ratings.dropna(subset=["User-ID", "ISBN", "Book-Rating"])
    return books, ratings


def filter_explicit(ratings: pd.DataFrame) -> pd.DataFrame:
    return ratings[ratings["Book-Rating"] > 0].copy()


def sparsity_filter(ratings_explicit: pd.DataFrame, min_user_ratings: int = 10, min_item_ratings: int = 10) -> pd.DataFrame:
    user_counts = ratings_explicit["User-ID"].value_counts()
    item_counts = ratings_explicit["ISBN"].value_counts()

    active_users = user_counts[user_counts >= min_user_ratings].index
    popular_items = item_counts[item_counts >= min_item_ratings].index

    filtered = ratings_explicit[
        ratings_explicit["User-ID"].isin(active_users) &
        ratings_explicit["ISBN"].isin(popular_items)
    ].copy()
    return filtered


@dataclass
class Mappings:
    user_id_map: Dict[int, int]
    item_id_map: Dict[str, int]
    idx_to_user: Dict[int, int]
    idx_to_isbn: Dict[int, str]


def encode_mappings(filtered_ratings: pd.DataFrame) -> Tuple[pd.DataFrame, Mappings]:
    user_id_map = {int(uid): idx for idx, uid in enumerate(filtered_ratings["User-ID"].unique())}
    item_id_map = {str(isbn): idx for idx, isbn in enumerate(filtered_ratings["ISBN"].unique())}

    df = filtered_ratings.copy()
    df["user_idx"] = df["User-ID"].map(user_id_map)
    df["item_idx"] = df["ISBN"].map(item_id_map)

    idx_to_user = {idx: uid for uid, idx in user_id_map.items()}
    idx_to_isbn = {idx: isbn for isbn, idx in item_id_map.items()}

    # Verify no NaN
    assert df["user_idx"].isna().sum() == 0, "NaN in user_idx!"
    assert df["item_idx"].isna().sum() == 0, "NaN in item_idx!"

    return df, Mappings(user_id_map, item_id_map, idx_to_user, idx_to_isbn)
