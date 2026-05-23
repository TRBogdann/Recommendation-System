from .cleanup import clean_value

def extract_user_subreddits(interactions):
    return set(
        clean_value(subreddit).lower()
        for subreddit in interactions["subreddit"].tolist()
        if clean_value(subreddit)
    )

def get_subreddit_bonus(post_subreddit, user_subreddits):
    post_subreddit = clean_value(post_subreddit).lower()

    if post_subreddit in user_subreddits:
        return 0.15
    
    return 0.0

def get_same_subreddit_candidate_indices(posts_df, user_subreddits, max_per_subreddit=100):
    candidate_indices = []

    for subreddit in user_subreddits:
        matching_rows = posts_df[
            posts_df["subreddit"].fillna("").str.lower() == subreddit
        ]

        if matching_rows.empty:
            continue

        candidate_indices.extend(matching_rows.index.tolist()[:max_per_subreddit])
    
    return candidate_indices
