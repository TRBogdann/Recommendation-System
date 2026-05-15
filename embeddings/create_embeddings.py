import sqlite3
import re
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


DB_PATH = "app.db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def clean_text(value):
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.lower() in ["nan", "none", "null", "[deleted]", "[removed]"]:
        return ""

    text = text.lower()

    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\*\*|\*|_|`|>", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def build_embedding_text(row):
    title = clean_text(row["title"])
    subreddit = clean_text(row["subreddit"])
    body = clean_text(row["body"])

    return f"""
Title: {title}
Subreddit: {subreddit}
Content: {body}
""".strip()


def main():
    print("Reading posts from app.db...")

    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query("""
        SELECT 
            id,
            title,
            subreddit,
            body,
            user_id,
            recommend_ready
        FROM posts
    """, conn)

    conn.close()




    df["title_clean"] = df["title"].apply(clean_text)
    df["subreddit_clean"] = df["subreddit"].apply(clean_text)
    df["body_clean"] = df["body"].apply(clean_text)

    df["embedding_text"] = df.apply(build_embedding_text, axis=1)

    df["useful_text"] = (
        df["title_clean"] + " " +
        df["subreddit_clean"] + " " +
        df["body_clean"]
    ).str.strip()

    df = df[df["useful_text"] != ""].copy()

    df = df.reset_index(drop=True)




    print("\nLoading Hugging Face model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Generating embeddings...")
    embeddings = model.encode(
        df["embedding_text"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print("Shape embeddings:", embeddings.shape)


    np.save("post_embeddings.npy", embeddings)
    df.to_csv("reddit_posts_metadata.csv", index=False)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, "reddit_posts.index")




if __name__ == "__main__":
    main()