# pip install exa-py
from exa_py import Exa
from dotenv import load_dotenv
import os

load_dotenv()   

exa = Exa(os.getenv('EXA_API_KEY'))

results = exa.get_contents(
    urls=["https://www.aven.com/support"],
    text={
        "includeHtmlTags": True
    },
    livecrawl="preferred", # preferring livecrawl over cache fallback
    subpages=1,
    summary={
        "query": "What does this page contain?"
    },
    subpage_target=["FAQ", "Questions", "Support"],
    extras={
        "links": 2,
        "image_links": 1
    }
)

import json

# Print results to console
print(results.results[0].summary)

# Write results to a markdown file
with open("results.md", "w", encoding="utf-8") as f:
    f.write(str(results))