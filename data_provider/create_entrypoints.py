import requests
import time
import random

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:150.0) Gecko/20100101 Firefox/150.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

session = requests.Session()
session.headers.update(headers)


retry_list = []
post_links = []
rejected_requets_count = 0
start = 0

with open("output/subreddits.txt", "r") as f_in, open("output/entrypoints.txt", "a") as f_out:
    for line in f_in:
        url = "https://www.reddit.com/" + line.strip() + ".json"
        params = {"limit": 100, "after": None}

        start += 1
        if start < 1691:
            continue 

        try:
            response = session.get(url, params=params, timeout=10)
            print(f"Status: {response.status_code} | URL: {url}")

            if response.status_code == 429:
                retry_list.append(url)
                rejected_requets_count += 1
                time_to_sleep = 300
                print(f'Rate limited! Pausing for {time_to_sleep}s...')
                time.sleep(time_to_sleep)
                continue

            if response.status_code == 200:
                rejected_requets_count = 0
                data = response.json()
                posts = data["data"]["children"]
                for post in posts:
                    d = post["data"]
                    link = "https://reddit.com" + d["permalink"]
                    post_links.append(link)
                    f_out.write(link + "\n")

        except Exception as e:
            print(f"Error: {e}")
            rejected_requets_count += 1
            time_to_sleep = 300
            print(f'Pausing for {time_to_sleep}s...')
            time.sleep(time_to_sleep)

        time.sleep(random.uniform(1, 3))

print(retry_list)