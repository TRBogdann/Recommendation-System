import requests
import sqlite3
import time
import random

# ── Constants ────────────────────────────────────────────────────────────────

PASSWORD_HASH = '$2a$12$bODggySyW.KzLdpCt/1BouCe0vpCxeDWZSvPHn6IHL.k5SsGzdtvq'

MIN_SLEEP = 0.3
MAX_SLEEP = 1.0

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:150.0) Gecko/20100101 Firefox/150.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# ── Session ──────────────────────────────────────────────────────────────────

session = requests.Session()
session.headers.update(HEADERS)

# ── DB: app.db ────────────────────────────────────────────────────────────────

app_conn = sqlite3.connect("app.db")
app_cursor = app_conn.cursor()
app_cursor.execute("PRAGMA foreign_keys = ON;")

app_cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id       INTEGER PRIMARY KEY,
        username VARCHAR(255),
        password VARCHAR(255)
    )
""")
app_cursor.execute("""
    CREATE TABLE IF NOT EXISTS posts (
        id             INTEGER PRIMARY KEY,
        title          VARCHAR(255),
        subreddit      VARCHAR(255),
        body           TEXT,
        user_id        INTEGER NOT NULL,
        recommend_ready BOOLEAN,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
""")
app_cursor.execute("""
    CREATE TABLE IF NOT EXISTS interactions (
        id      INTEGER PRIMARY KEY,
        post_id INTEGER,
        user_id INTEGER,
        comment TEXT,
        type    VARCHAR(100),
        FOREIGN KEY (post_id) REFERENCES posts(id),
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
""")
app_conn.commit()

# ── DB: crawler.db ────────────────────────────────────────────────────────────

crawler_conn = sqlite3.connect("crawler.db")
crawler_conn.row_factory = sqlite3.Row

# ── Helpers ───────────────────────────────────────────────────────────────────

user_cache: dict[str, int] = {}

def jitter():
    time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))


def get_or_create_user(username: str) -> int:
    if username in user_cache:
        return user_cache[username]
    row = app_cursor.execute(
        "SELECT id FROM users WHERE username = ?", (username,)
    ).fetchone()
    if row:
        user_cache[username] = row[0]
        return row[0]
    app_cursor.execute(
        "INSERT INTO users (username, password) VALUES (?, ?)",
        (username, PASSWORD_HASH),
    )
    uid = app_cursor.lastrowid
    user_cache[username] = uid
    return uid


def collect_comments(comment_list: list, post_db_id: int, depth: int = 0) -> int:
    """Recursively insert all comments; returns count inserted."""
    count = 0
    for item in comment_list:
        if not isinstance(item, dict):
            continue
        kind = item.get("kind")
        data = item.get("data", {})

        if kind == "more":
            continue  # would need extra API calls to expand
        if kind != "t1":
            continue

        author = data.get("author", "[deleted]")
        body   = data.get("body", "")

        if author and author not in ("[deleted]", "AutoModerator"):
            uid = get_or_create_user(author)
            app_cursor.execute(
                "INSERT INTO interactions (post_id, user_id, comment, type) VALUES (?, ?, ?, ?)",
                (post_db_id, uid, body, "comment"),
            )
            count += 1

        replies = data.get("replies")
        if isinstance(replies, dict):
            children = replies.get("data", {}).get("children", [])
            count += collect_comments(children, post_db_id, depth + 1)

    return count


def process_post(url: str):
    print(f"\nFetching: {url}")
    resp = session.get(url, timeout=10)
    resp.raise_for_status()

    payload = resp.json()

    post_data        = payload[0]["data"]["children"][0]["data"]
    comments_listing = payload[1]["data"]["children"]

    title     = post_data.get("title", "")
    subreddit = post_data.get("subreddit", "")
    body      = post_data.get("selftext", "")
    author    = post_data.get("author", "[deleted]")
    reddit_id = post_data.get("id", "")

    print(f"  Post  : {title[:60]!r}")
    print(f"  Author: {author}  r/{subreddit}")

    uid         = get_or_create_user(author)
    post_int_id = int(reddit_id, 36)

    existing = app_cursor.execute(
        "SELECT id FROM posts WHERE id = ?", (post_int_id,)
    ).fetchone()

    if existing:
        print(f"  Already in DB (id={post_int_id}), skipping insert.")
    else:
        app_cursor.execute(
            "INSERT INTO posts (id, title, subreddit, body, user_id, recommend_ready) VALUES (?, ?, ?, ?, ?, ?)",
            (post_int_id, title, subreddit, body, uid, False),
        )
        print(f"  Inserted post id={post_int_id}")

    n = collect_comments(comments_listing, post_int_id)
    print(f"  Inserted {n} comments")
    app_conn.commit()


# ── Main loop ─────────────────────────────────────────────────────────────────

crawler_cursor = crawler_conn.execute("SELECT * FROM links WHERE visited = 0")
rows = crawler_cursor.fetchall()

for row in rows:
    url = row["url"].rstrip("/") + ".json"
    try:
        process_post(url)
        crawler_conn.execute("UPDATE links SET visited = 1 WHERE id = ?", (row["id"],))
        crawler_conn.commit()
    except requests.HTTPError as err:
        status = err.response.status_code
        print(f"  HTTP {status} for {url}")
        if status == 403:
            jitter()
        elif status == 429:
            print("  Rate limited — pausing 5 minutes")
            time.sleep(300)
    finally:
        jitter()  # polite delay after every request regardless

app_conn.close()
crawler_conn.close()