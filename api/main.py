from fastapi import FastAPI, HTTPException, Header, Depends
import os
from starlette.concurrency import run_in_threadpool

from .schemas import RecommendRequest, RecommendResponse
from .recommend import recommend_posts_for_user


API_KEY = os.getenv('API_KEY')
def verify_api_key(x_api_key: str = Header(...)):
    if API_KEY is None:
        raise HTTPException(status_code=500, detail="API key not configured on server.")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return True

app = FastAPI(title="Reddit Recommendation API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse, dependencies=[Depends(verify_api_key)])
async def recommend(req: RecommendRequest):
    username, results = await run_in_threadpool(recommend_posts_for_user, req.user_id, req.top_x)

    if results is None or len(results) == 0:
        raise HTTPException(status_code=404, detail="No recommendations found for given user_id")

    return RecommendResponse(
        user_id=req.user_id,
        username=username,
        top_x=req.top_x,
        recommended_posts=results
    )
