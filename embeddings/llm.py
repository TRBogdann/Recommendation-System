from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """
You are a recommendation explainability assistant.
Explain briefly and naturally why a Reddit post was recommended.
Use only the provided context.
""".strip()

client = InferenceClient(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    token=os.getenv("HF_TOKEN"),
    provider="auto"
)

def get_llm_response(prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    response = client.chat_completion(
        messages=messages,
        max_tokens=120,
        temperature=0.4
    )

    return response.choices[0].message["content"]


def generate_recommendation_explanation(
    user_keywords,
    user_subreddits,
    recommended_title,
    recommended_subreddit,
    semantic_score,
    collaborative_score
):
    keywords_text = ", ".join(user_keywords[:5])
    subreddits_text = ", ".join(list(user_subreddits)[:5])

    prompt = f"""
User interests:
- Keywords: {keywords_text}
- Frequent subreddits: {subreddits_text}

Recommended post:
Title: {recommended_title}
Subreddit: {recommended_subreddit}

Recommendation signals:
- Semantic similarity score: {semantic_score:.4f}
- Collaborative filtering score: {collaborative_score:.4f}

Explain in 1-2 concise sentences why this recommendation is relevant.
"""
    response = get_llm_response(prompt)
    print("LLM response:", response)
    return response