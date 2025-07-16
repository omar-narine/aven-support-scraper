from firecrawl import FirecrawlApp, ScrapeOptions
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import json

load_dotenv()

app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

crawl_status = app.crawl_url(
  'https://www.aven.com/', 
  limit=100, 
  scrape_options=ScrapeOptions(formats=['markdown'], maxAge=3600000),
  poll_interval=30, 
  crawl_entire_domain=True,
  allow_subdomains=True,
)
print(crawl_status) 


# crawl_status = app.crawl_url(
#   'https://docs.vapi.ai/quickstart/web', 
#   limit=1, 
#   scrape_options=ScrapeOptions(formats=['markdown']),
#   poll_interval=30, 
#   crawl_entire_domain=False,  # Changed to False for single page
#   allow_subdomains=False,     # Changed to False for single page
# )
print(f"Crawl status: {crawl_status.status}")

with open("crawl_result.json", "w", encoding="utf-8") as f:
    crawl_dict = crawl_status.model_dump() if hasattr(crawl_status, 'model_dump') else vars(crawl_status)
    json.dump(crawl_dict, f, indent=4, ensure_ascii=False, default=str)

