from typing import List, Optional
from pydantic import BaseModel


class RecommendRequest(BaseModel):
    user_id: str
    top_x: Optional[int] = 10


class Post(BaseModel):
    rank: int
    id: str
    title: str
    subreddit: str
    body: str
    semantic_score: float
    collaborative_score: float
    subreddit_bonus: float
    keyword_bonus: float
    post_sentiment: str
    post_sentiment_confidence: float
    user_dominant_sentiment: str
    sentiment_bonus: float
    final_score: float
    explanation: Optional[str] = None


class RecommendResponse(BaseModel):
    user_id: str
    username: Optional[str]
    top_x: int
    recommended_posts: List[Post]
