import requests

url = "https://app.social-rise.com/api/subreddits"
n_of_iter = 200

def get_params(page):
    return {
        "page": page,
        "per_page": 250,
        "sort_by": "subscribers",
        "sort_dir": "desc",
        "filters[0][field]": "status",
        "filters[0][operator]": "<>",
        "filters[0][value]": "noexist",
        "filters[0][type]": 1,
        "filters[1][field]": "nsfw",
        "filters[1][operator]": "=",
        "filters[1][value]": 0,
        "filters[1][type]": 2,
        "filters[2][field]": "name",
        "filters[2][operator]": "not like",
        "filters[2][value]": r"u\_%",
        "filters[2][type]": 1,
        "filters[3][field]": "status",
        "filters[3][operator]": "<>",
        "filters[3][value]": "banned",
        "filters[3][type]": 1,
        "search_fields[]": "name",
        "search_value": "",
    }

with open("output/subreddits.txt", "w") as f:
    page = 1
    while page <= n_of_iter:
        response = requests.get(url, params=get_params(page))
        data = response.json()
        items = data["data"]
        if not items:
            break
        for item in items:
            f.write('r/' + item["name"] + "\n")
        print(f"Page {page} done ({len(items)} subreddits)")
        page += 1