"""
Morning Brief — Daily AI Newspaper Generator
Runs via GitHub Actions every morning at 06:00 Israel time.
Uses Gemini 1.5 Flash (free tier) to analyze and write articles.
"""

import os
import json
import time
import feedparser
import requests
import google.generativeai as genai
from datetime import datetime, timezone, timedelta
from dateutil import parser as dateparser
from pathlib import Path


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

ISRAEL_TZ = timezone(timedelta(hours=3))

RSS_FEEDS = {
    "geopolitics": [
        {"name": "The Economist — World", "url": "https://www.economist.com/the-world-this-week/rss.xml"},
        {"name": "Foreign Policy", "url": "https://foreignpolicy.com/feed/"},
        {"name": "Reuters — World", "url": "https://feeds.reuters.com/reuters/worldNews"},
    ],
    "economy": [
        {"name": "The Economist — Finance", "url": "https://www.economist.com/finance-and-economics/rss.xml"},
        {"name": "Politico Economy", "url": "https://rss.politico.com/economy.xml"},
        {"name": "Reuters — Business", "url": "https://feeds.reuters.com/reuters/businessNews"},
    ],
    "technology": [
        {"name": "The Economist — Technology", "url": "https://www.economist.com/science-and-technology/rss.xml"},
        {"name": "MIT Technology Review", "url": "https://www.technologyreview.com/feed/"},
        {"name": "VentureBeat — AI", "url": "https://venturebeat.com/category/ai/feed/"},
    ],
}

ARTICLES_PER_CATEGORY = 3
SUMMARY_WORDS = 80
DEEP_DIVE_WORDS = 220


def fetch_articles(category, feeds):
    articles = []
    cutoff = datetime.now(tz=ISRAEL_TZ) - timedelta(hours=30)

    for feed_info in feeds:
        try:
            print(f"  Fetching: {feed_info['name']}")
            feed = feedparser.parse(feed_info["url"])

            for entry in feed.entries[:6]:
                pub_date = None
                if hasattr(entry, "published"):
                    try:
                        pub_date = dateparser.parse(entry.published)
                        if pub_date and pub_date.tzinfo is None:
                            pub_date = pub_date.replace(tzinfo=timezone.utc)
                    except Exception:
                        pass

                if pub_date and pub_date < cutoff:
                    continue

                title = entry.get("title", "").strip()
                summary = entry.get("summary", entry.get("description", "")).strip()
                link = entry.get("link", "")

                if title and summary and len(summary) > 60:
                    articles.append({
                        "title": title,
                        "summary": summary[:1200],
                        "link": link,
                        "source": feed_info["name"],
                        "published": pub_date.isoformat() if pub_date else "",
                        "category": category,
                    })

                if len(articles) >= ARTICLES_PER_CATEGORY * len(feeds):
                    break

        except Exception as e:
            print(f"  Warning: Could not fetch {feed_info['name']}: {e}")

    seen_titles = set()
    unique = []
    for a in articles:
        key = a["title"][:50].lower()
        if key not in seen_titles:
            seen_titles.add(key)
            unique.append(a)

    return unique[:ARTICLES_PER_CATEGORY]


def analyze_with_gemini(article):
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""You are the editor of a prestigious daily newspaper for an inte