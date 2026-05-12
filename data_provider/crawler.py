import requests
import sqlite3
import random
import time
import re

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:138.0) Gecko/20100101 Firefox/138.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Priority": "u=0, i",
    "TE": "trailers",
}

MIN_SLEEP = 0.5
MAX_SLEEP = 1.5
session = requests.Session()
session.headers.update(headers)

def jitter():
    time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))

def extract_subreddit(url: str) -> str:
    match = re.search(r'reddit\.com/r/([^/]+)', url)
    return 'r/' + match.group(1) if match else ""

def get_user_post_links(username_link):
    all_links = set()

    params = {"limit": 100}
    response = session.get(username_link, params=params, timeout=10)
    if response.status_code != 200:
        raise Exception(response.status_code)

    data = response.json()

    children = data["data"]["children"]

    for item in children:
        kind = item["kind"]
        d = item["data"]

        if kind == "t3":
            link = f"https://www.reddit.com{d['permalink']}"
            all_links.add(link)

        elif kind == "t1":
            post_url = f"https://www.reddit.com/r/{d['subreddit']}/comments/{d['link_id'][3:]}/"
            all_links.add(post_url)

    return all_links

def extract_authors(comments):
    authors = set()
    for comment in comments:
        if comment["kind"] == "t1":
            d = comment["data"]
            author = d.get("author")
            if author and author != "[deleted]":
                authors.add('https://www.reddit.com/user/' + author + '.json')

            replies = d.get("replies")
            if replies and isinstance(replies, dict):
                children = replies.get("data", {}).get("children", [])
                authors.update(extract_authors(children))
    return authors

def get_users(url):
    response = response = session.get(url, timeout=10)
    if response.status_code != 200:
        raise Exception(response.status_code)
    data = response.json()

    all_users = set()

    post_author = data[0]["data"]["children"][0]["data"].get("author")
    if post_author:
        all_users.add('https://www.reddit.com/user/' + post_author + '.json')

    comment_children = data[1]["data"]["children"]
    all_users.update(extract_authors(comment_children))

    return all_users

conn = sqlite3.connect("crawler.db")
damping_factor = 0.85

conn.execute("""
    CREATE TABLE IF NOT EXISTS links (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        url        TEXT UNIQUE,
        subreddit  TEXT,
        visited    NUMBER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")
conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_url ON links(url)")
conn.commit()

with open("output/entrypoints.txt", "r") as f:
    entry_points = [line.strip() for line in f.readlines()]

start = entry_points[random.randint(0, len(entry_points) - 1)]
entry_points.remove(start)

def crawl(current, entry_points):
    if current[-1] == '/':
        current = current[:-1] + '.json'

    urls = []
    try:
        users = get_users(current)
        jitter()
        for user in users:
            posts = list(get_user_post_links(user))
            urls += posts
            entry_points += posts
            rows = [(p, extract_subreddit(p)) for p in posts]
            conn.executemany(
                "INSERT OR IGNORE INTO links (url, subreddit) VALUES (?, ?)",
                rows
            )
            conn.commit()
            jitter()
        
            
    except Exception as err:
        if err.args[0] == 403:
            jitter()

        if err.args[0] == 429:
            print("Rate limited, pausing for 5 minutes")
            time.sleep(300)

    print(f"Crawler found {len(urls)} links")
    if len(urls) == 0:
        urls = entry_points
    else:
        prob = random.uniform(0, 1)
        urls = urls if prob <= damping_factor else entry_points
    
    new_start = urls[random.randint(0, len(urls) - 1)]
    entry_points = list(set(entry_points))
    entry_points.remove(new_start)
    return new_start
    
while True:
    start = crawl(start, entry_points)