from .cleanup import clean_value
from bson import ObjectId
import pandas as pd

from api.mongo_client import mongo_db


def load_user_interactions(user_id):
    user_id = ObjectId(user_id)
                       
    pipeline = [
        {"$match": {"user_id": user_id}},
        {"$lookup": {
            "from": "users",
            "localField": "user_id",
            "foreignField": "_id",
            "as": "user"
        }},
        {"$unwind": {"path": "$user", "preserveNullAndEmptyArrays": True}},
        {"$lookup": {
            "from": "posts",
            "localField": "post_id",
            "foreignField": "_id",
            "as": "post"
        }},
        {"$unwind": {"path": "$post", "preserveNullAndEmptyArrays": True}},
        {"$project": {
            "_id": 0,
            "user_id": 1,
            "username": "$user.username",
            "type": 1,
            "comment": 1,
            "post_id": "$post._id",
            "title": "$post.title",
            "subreddit": "$post.subreddit",
            "body": "$post.body",
        }},
        {"$limit": 100},
    ]

    interactions = pd.DataFrame(
        list(mongo_db.interactions.aggregate(pipeline))
    )

    return interactions

def build_user_profile_text(interactions, user_keywords=None, user_subreddits=None):
    if interactions.empty:
        return None
    
    username = interactions["username"].iloc[0]

    comments = " ".join(
        clean_value(comment)
        for comment in interactions["comment"].tolist()
    )

    titles = " ".join(
        clean_value(title)
        for title in interactions["title"].dropna().unique().tolist()
    )

    subreddits_text = ", ".join(sorted(user_subreddits)) if user_subreddits else ""
    keywords_text = ", ".join(user_keywords) if user_keywords else ""
    profile_text = f"""
        User profile generated from Reddit interactions.
        Username: {username}

        The user frequently interacts with these subreddits:
        {subreddits_text}

        Important terms automatically extracted from the user's activity:
        {keywords_text}

        The user's comments:
        {comments}

        Titles of posts the user interacted with:
        {titles}

        Recommend Reddit posts that match the user's interests, communities, terminology, topics, and previous activity.
        """.strip()
    return profile_text
