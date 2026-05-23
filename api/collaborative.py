import os

from dotenv import load_dotenv
import numpy as np
import pandas as pd

from api.mongo_client import mongo_db

load_dotenv()

USE_COLLABORATIVE = (
    os.getenv("USE_COLLABORATIVE", "false").lower() == "true"
)


_similarity_df = None
_interaction_matrix_cache = None
_post_mapping_cache = None

def get_interaction_matrix():
    global _interaction_matrix_cache, _post_mapping_cache

    if _interaction_matrix_cache is not None:
        return _interaction_matrix_cache, _post_mapping_cache

    from scipy.sparse import csr_matrix


    interactions_df = pd.DataFrame(
        list(mongo_db.interactions.find({}, {"_id": 0, "user_id": 1, "post_id": 1}))
    )

    interactions_df["interaction"] = 1

    user_ids = interactions_df["user_id"].astype("category")
    post_ids = interactions_df["post_id"].astype("category")

    matrix = csr_matrix(
        (
            interactions_df["interaction"],
            (user_ids.cat.codes, post_ids.cat.codes)
        )
    )

    _interaction_matrix_cache = matrix
    _post_mapping_cache = post_ids.cat.categories

    return matrix, _post_mapping_cache


def get_similarity_df():
    global _similarity_df

    if not USE_COLLABORATIVE: 
        return None

    if _similarity_df is not None:
        return _similarity_df

    interaction_matrix, post_mapping = get_interaction_matrix()

    item_matrix = interaction_matrix.T

    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(item_matrix)

    _similarity_df = pd.DataFrame(
        similarity_matrix,
        index=post_mapping,
        columns=post_mapping
    )

    return _similarity_df

def get_collaborative_score(
    candidate_post_id,
    interacted_post_ids,
    similarity_df=None
):
    if not USE_COLLABORATIVE: 
        return 0.0 
    
    if similarity_df is None: 
        return 0.0
    
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

def is_collaborative_enabled(): 
    return USE_COLLABORATIVE