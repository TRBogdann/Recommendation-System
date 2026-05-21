from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
import sqlite3

def build_interaction_matrix(db_path):
    conn = sqlite3.connect(db_path)

    interactions_df = pd.read_sql_query("""
        SELECT user_id, post_id
        FROM interactions
    """, conn)

    conn.close()

    interactions_df["interaction"] = 1

    user_ids = interactions_df["user_id"].astype("category")
    post_ids = interactions_df["post_id"].astype("category")

    sparse_matrix = csr_matrix(
        (
            interactions_df["interaction"],
            (user_ids.cat.codes, post_ids.cat.codes)
        )
    )

    return (
        sparse_matrix,
        user_ids.cat.categories,
        post_ids.cat.categories
    )


def build_item_similarity_matrix(sparse_matrix, post_mapping):
    item_matrix = sparse_matrix.T

    similarity_matrix = cosine_similarity(item_matrix)

    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=post_mapping,
        columns=post_mapping
    )

    return similarity_df

def get_collaborative_score(
    candidate_post_id,
    interacted_post_ids,
    similarity_df
):
    if candidate_post_id not in similarity_df.index:
        return 0.0

    similarities = []

    for interacted_post_id in interacted_post_ids:
        if interacted_post_id not in similarity_df.index:
            continue

        similarity = similarity_df.loc[
            candidate_post_id,
            interacted_post_id
        ]

        similarities.append(similarity)

    if not similarities:
        return 0.0

    return float(np.mean(similarities))
