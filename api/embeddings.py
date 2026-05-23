import os
import faiss
import numpy as np
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd

load_dotenv()

USE_HF_EMBEDDINGS = (
    os.getenv("USE_HF_EMBEDDINGS", "false").lower() == "true"
)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BASE_DIR = Path(__file__).resolve().parent
EMBEDDINGS_DIR = BASE_DIR / "data"

_FAISS_INDEX = None
_POST_EMBEDDINGS = None
_posts_df = None
_embedding_model = None
_hf_client = None



def get_faiss_index():
    global _FAISS_INDEX

    if _FAISS_INDEX is None:
        _FAISS_INDEX = faiss.read_index(str(EMBEDDINGS_DIR / "reddit_posts.index"))

    return _FAISS_INDEX


def get_post_embeddings():
    global _POST_EMBEDDINGS

    if _POST_EMBEDDINGS is None:
        _POST_EMBEDDINGS = np.load(str(EMBEDDINGS_DIR / "post_embeddings.npy"))

    return _POST_EMBEDDINGS

def get_posts_df():
    global _posts_df
    if _posts_df is None:
        _posts_df = pd.read_csv(str(EMBEDDINGS_DIR / "reddit_posts_metadata.csv"))
    return _posts_df


def get_embedding_model():
    global _embedding_model

    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

    return _embedding_model


def get_hf_client():
    global _hf_client

    if _hf_client is None:
        from huggingface_hub import InferenceClient

        _hf_client = InferenceClient(
            token=os.getenv("HF_TOKEN")
        )

    return _hf_client


def get_embedding(text: str):
    if USE_HF_EMBEDDINGS:
        print("Using Hugging Face embeddings model")
        client = get_hf_client()

        embedding = client.feature_extraction(
            text,
            model=EMBEDDING_MODEL_NAME
        )

        embedding = np.array(
            embedding,
            dtype=np.float32
        )

    else:
        model = get_embedding_model()

        embedding = model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    embedding = embedding / np.linalg.norm(embedding)

    embedding = embedding.reshape(1, -1)

    return embedding
