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
        {"name": "The Economist World", "url": "https://www.economist.com/the-world-this-week/rss.xml"},
        {"name": "Foreign Policy", "url": "https://foreignpolicy.com/feed/"},
        {"name": "Reuters World", "url": "https://feeds.reuters.com/reuters/worldNews"},
    ],
    "economy": [
        {"name": "The Economist Finance", "url": "https://www.economist.com/finance-and-economics/rss.xml"},
        {"name": "Politico Economy", "url": "https://rss.politico.com/economy.xml"},
        {"name": "Reuters Business", "url": "https://feeds.reuters.com/reuters/businessNews"},
    ],
    "technology": [
        {"name": "The Economist Technology", "url": "https://www.economist.com/science-and-technology/rss.xml"},
        {"name": "MIT Technology Review", "url": "https://www.technologyreview.com/feed/"},
        {"name": "VentureBeat AI", "url": "https://venturebeat.com/category/ai/feed/"},
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
            print("  Fetching: " + feed_info["name"])
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
            print("  Warning: " + str(e))
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
    prompt = (
        "You are the editor of a prestigious daily newspaper.\n"
        "Analyze this article and return ONLY valid JSON, no markdown, no backticks.\n\n"
        "ARTICLE TITLE: " + article["title"] + "\n"
        "SOURCE: " + article["source"] + "\n"
        "CONTENT: " + article["summary"] + "\n\n"
        "Return this exact JSON structure:\n"
        '{\n'
        '  "headline": "sharp headline max 12 words",\n'
        '  "deck": "one sentence subheadline max 20 words",\n'
        '  "summary": "80-word summary of key facts",\n'
        '  "deep_dive": "220-word analytical deep-dive",\n'
        '  "vocab": [\n'
        '    {"word": "word1", "hebrew": "translation1"},\n'
        '    {"word": "word2", "hebrew": "translation2"}\n'
        '  ]\n'
        '}'
    )
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()
        data = json.loads(text)
        return {
            "headline": data.get("headline", article["title"]),
            "deck": data.get("deck", ""),
            "summary": data.get("summary", article["summary"][:300]),
            "deep_dive": data.get("deep_dive", ""),
            "vocab": data.get("vocab", []),
        }
    except Exception as e:
        print("  Gemini error: " + str(e))
        return {
            "headline": article["title"],
            "deck": "",
            "summary": article["summary"][:400],
            "deep_dive": "",
            "vocab": [],
        }


def apply_vocab_tooltips(text, vocab):
    for item in vocab:
        word = item.get("word", "")
        he = item.get("hebrew", "")
        if word and he and word.lower() in text.lower():
            idx = text.lower().find(word.lower())
            original = text[idx: idx + len(word)]
            replacement = '<span class="vocab-word" data-he="' + he + '">' + original + '</span>'
            text = text[:idx] + replacement + text[idx + len(word):]
    return text


def build_article_html(article, ai, article_id, is_main=False):
    body_text = apply_vocab_tooltips(ai["summary"], ai.get("vocab", []))
    deep_text = apply_vocab_tooltips(ai["deep_dive"], ai.get("vocab", []))
    category_display = {
        "geopolitics": "Geopolitics",
        "economy": "Economy",
        "technology": "Technology / AI",
    }.get(article["category"], article["category"].title())
    pull_quote = ""
    if is_main and ai.get("deck"):
        pull_quote = '<div class="pull-quote">"' + ai["deck"] + '"</div>'
    source_link = ""
    if article.get("link"):
        source_link = '<a href="' + article["link"] + '" target="_blank" rel="noopener" class="source-link">Read original</a>'
    return (
        '<div class="article" id="' + article_id + '" data-cat="' + article["category"] + '">'
        '<div class="article-source">' + category_display + " · " + article["source"] + "</div>"
        '<div class="article-headline">' + ai["headline"] + "</div>"
        '<div class="article-deck">' + ai["deck"] + "</div>"
        + pull_quote +
        '<div class="article-body"><p class="drop-cap">' + body_text + "</p></div>"
        '<button class="expand-btn" onclick="toggleExpand(this, \'exp_' + article_id + '\')">+ Deep dive</button>'
        '<div class="expanded-content" id="exp_' + article_id + '"><p>' + deep_text + "</p>" + source_link + "</div>"
        '<div class="article-actions">'
        '<button class="mark-read-btn" onclick="markRead(this, \'' + article_id + '\')">&#10003; Mark as read</button>'
        '<div class="interest-wrap">'
        '<span class="interest-label">Interest</span>'
        '<input type="range" min="1" max="5" value="3" step="1" data-id="' + article_id + '" oninput="updateSlider(this)">'
        '<span class="interest-val">3</span>'
        "</div></div></div>"
    )


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>The Morning Brief - {date_display}</title>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;0,900;1,400&family=Source+Serif+4:ital,opsz,wght@0,8..60,300;0,8..60,400&family=Inter:wght@300;400;500&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{--ink:#1a1a18;--ink-muted:#6b6b64;--ink-faint:#a8a89e;--paper:#faf8f3;--paper-warm:#f2efe6;--border:#ccc9be;--green:#1D9E75;--green-bg:#E1F5EE;--serif:'Playfair Display',Georgia,serif;--body:'Source Serif 4',Georgia,serif;--sans:'Inter',system-ui,sans-serif}}
body{{background:var(--paper);color:var(--ink);font-family:var(--body);font-size:14px;line-height:1.7}}
.paper-header{{border-top:5px solid var(--ink);border-bottom:1px solid var(--ink);padding:10px 24px 12px;text-align:center}}
.paper-meta{{display:flex;justify-content:space-between;font-family:var(--sans);font-size:10px;font-weight:300;color:var(--ink-muted);letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px}}
.paper-title{{font-family:var(--serif);font-size:clamp(32px,6vw,52px);font-weight:900;letter-spacing:-.02em;line-height:1;margin:4px 0}}
.paper-tagline{{font-family:var(--sans);font-size:10px;font-weight:300;letter-spacing:.18em;color:var(--ink-muted);text-transform:uppercase;margin-top:5px}}
.progress-bar-wrap{{background:var(--paper-warm);border-bottom:.5px solid var(--border);padding:7px 24px;display:flex;align-items:center;gap:12px;font-family:var(--sans);font-size:11px;color:var(--ink-muted)}}
.progress-bar{{flex:1;height:3px;background:var(--border);border-radius:2px;overflow:hidden}}
.progress-fill{{height:100%;background:var(--ink);border-radius:2px;width:0%;transition:width .4s ease}}
.section-nav{{display:flex;border-top:1px solid var(--ink);border-bottom:3px double var(--border);padding:0 24px;overflow-x:auto}}
.nav-tab{{font-family:var(--sans);font-size:10px;font-weight:500;letter-spacing:.12em;text-transform:uppercase;padding:6px 14px;cursor:pointer;color:var(--ink-muted);border:none;background:none;border-right:.5px solid var(--border);transition:all .15s;white-space:nowrap}}
.nav-tab:hover{{background:var(--paper-warm)}}
.nav-tab.active{{background:var(--ink);color:var(--paper)}}
.nav-time{{font-family:var(--sans);font-size:10px;color:var(--ink-muted);padding:6px 14px;margin-left:auto}}
.columns{{display:grid;grid-template-columns:1fr 1.65fr 1fr;max-width:1200px;margin:0 auto;padding:0 16px}}
.col{{padding:16px 18px;border-right:.5px solid var(--border)}}
.col:last-child{{border-right:none}}
.article{{margin-bottom:24px;padding-bottom:20px;border-bottom:.5px solid var(--border);transition:opacity .3s}}
.article:last-child{{border-bottom:none;margin-bottom:0;padding-bottom:0}}
.article.is-read{{opacity:.4}}
.article-source{{font-family:var(--sans);font-size:9px;font-weight:500;letter-spacing:.14em;text-transform:uppercase;color:var(--ink-muted);margin-bottom:5px}}
.article-headline{{font-family:var(--serif);font-size:19px;font-weight:700;line-height:1.2;margin-bottom:6px}}
.col:nth-child(2) .article-headline{{font-size:26px}}
.article-deck{{font-family:var(--sans);font-size:11px;font-weight:300;font-style:italic;color:var(--ink-muted);margin-bottom:10px;line-height:1.5}}
.article-body{{font-size:13.5px;line-height:1.7;font-weight:300}}
.article-body p{{margin-bottom:8px}}
.drop-cap::first-letter{{font-family:var(--serif);font-size:52px;font-weight:900;float:left;line-height:.82;margin:4px 6px 0 0}}
.pull-quote{{border-top:2px solid var(--ink);border-bottom:2px solid var(--ink);margin:14px 0;padding:10px 6px;font-family:var(--serif);font-size:14px;font-style:italic;line-height:1.45;text-align:center}}
.vocab-word{{border-bottom:1px dotted var(--ink-muted);cursor:help;position:relative}}
.vocab-word::after{{content:attr(data-he);position:absolute;bottom:calc(100% + 4px);left:50%;transform:translateX(-50%);background:var(--ink);color:var(--paper);padding:3px 9px;border-radius:3px;font-size:11px;font-family:var(--sans);white-space:nowrap;opacity:0;pointer-events:none;transition:opacity .15s;z-index:100}}
.vocab-word:hover::after{{opacity:1}}
.expand-btn{{font-family:var(--sans);font-size:10px;font-weight:500;letter-spacing:.08em;text-transform:uppercase;background:none;border:.5px solid var(--border);color:var(--ink-muted);padding:5px 11px;cursor:pointer;border-radius:2px;margin-top:10px;transition:all .15s}}
.expand-btn:hover{{background:var(--paper-warm)}}
.expanded-content{{display:none;margin-top:10px;padding-top:10px;border-top:.5px dashed var(--border);font-size:13px;line-height:1.7;color:var(--ink-muted)}}
.article-actions{{display:flex;align-items:center;gap:10px;margin-top:12px;flex-wrap:wrap}}
.mark-read-btn{{font-family:var(--sans);font-size:9px;font-weight:500;letter-spacing:.1em;text-transform:uppercase;background:none;border:.5px solid var(--border);color:var(--ink-muted);padding:5px 10px;cursor:pointer;border-radius:2px;transition:all .15s}}
.mark-read-btn.done{{border-color:var(--green);color:var(--green);background:var(--green-bg)}}
.interest-wrap{{display:flex;align-items:center;gap:7px;flex:1;min-width:110px}}
.interest-label{{font-family:var(--sans);font-size:9px;color:var(--ink-faint);white-space:nowrap;letter-spacing:.08em;text-transform:uppercase}}
input[type=range]{{flex:1;height:2px;accent-color:var(--ink);cursor:pointer}}
.interest-val{{font-family:var(--sans);font-size:10px;font-weight:500;min-width:14px;text-align:right;color:var(--ink-muted)}}
.source-link{{font-family:var(--sans);font-size:9px;color:var(--ink-faint);text-decoration:none;letter-spacing:.06em;text-transform:uppercase}}
.col-section-label{{font-family:var(--serif);font-size:10px;font-style:italic;text-align:center;color:var(--ink-muted);margin:-8px 0 18px;padding:0 12px;position:relative}}
.col-section-label::before,.col-section-label::after{{content:'';position:absolute;top:50%;width:24%;height:.5px;background:var(--border)}}
.col-section-label::before{{left:0}}
.col-section-label::after{{right:0}}
.paper-footer{{border-top:3px double var(--border);border-bottom:3px solid var(--ink);padding:8px 24px;display:flex;justify-content:space-between;font-family:var(--sans);font-size:9px;color:var(--ink-muted);letter-spacing:.09em;text-transform:uppercase;margin-top:8px}}
</style>
</head>
<body>
<header class="paper-header">
  <div class="paper-meta">
    <span>Geopolitics · Economy · Technology & AI</span>
    <span>{date_display}</span>
    <span>Automated Edition · Gemini AI</span>
  </div>
  <div class="paper-title">The Morning Brief</div>
  <div class="paper-tagline">Your personal newspaper — refreshed every morning at 06:00</div>
</header>
<div class="progress-bar-wrap">
  <span id="read-count">0 / {total_articles} read</span>
  <div class="progress-bar"><div class="progress-fill" id="prog-fill"></div></div>
  <span>~10 min read · Updated {update_time}</span>
</div>
<nav class="section-nav">
  <button class="nav-tab active" onclick="filterSection('all',this)">All</button>
  <button class="nav-tab" onclick="filterSection('geopolitics',this)">Geopolitics</button>
  <button class="nav-tab" onclick="filterSection('economy',this)">Economy</button>
  <button class="nav-tab" onclick="filterSection('technology',this)">Technology / AI</button>
  <span class="nav-time">{update_time} IST</span>
</nav>
<div class="columns">
  <div class="col">{col_left}</div>
  <div class="col">{col_center}</div>
  <div class="col">{col_right}</div>
</div>
<footer class="paper-footer">
  <span>The Morning Brief {year}</span>
  <span>Powered by Gemini AI · GitHub Actions · GitHub Pages</span>
  <span>Auto-generated {update_time} IST</span>
</footer>
<script>
const total={total_articles};
let readSet=new Set();
function updateProgress(){{var n=readSet.size;document.getElementById('read-count').textContent=n+' / '+total+' read';document.getElementById('prog-fill').style.width=Math.round((n/total)*100)+'%';}}
function markRead(btn,id){{var article=document.getElementById(id);if(readSet.has(id)){{readSet.delete(id);article.classList.remove('is-read');btn.innerHTML='&#10003; Mark as read';btn.classList.remove('done');}}else{{readSet.add(id);article.classList.add('is-read');btn.innerHTML='&#10003; Read';btn.classList.add('done');localStorage.setItem('read_'+id,'1');}}updateProgress();}}
function toggleExpand(btn,expId){{var el=document.getElementById(expId);var open=el.style.display==='block';el.style.display=open?'none':'block';btn.textContent=open?'+ Deep dive':'- Close';}}
function updateSlider(input){{input.nextElementSibling.textContent=input.value;localStorage.setItem('rating_'+input.dataset.id,input.value);}}
function filterSection(cat,btn){{document.querySelectorAll('.nav-tab').forEach(function(t){{t.classList.remove('active');}});btn.classList.add('active');document.querySelectorAll('.article').forEach(function(a){{a.style.display=(cat==='all'||a.dataset.cat===cat)?'':'none';}});}}
document.querySelectorAll('.article').forEach(function(a){{var id=a.id;if(localStorage.getItem('read_'+id)){{a.classList.add('is-read');readSet.add(id);var btn=a.querySelector('.mark-read-btn');if(btn){{btn.innerHTML='&#10003; Read';btn.classList.add('done');}}}}var slider=a.querySelector('input[type=range]');if(slider){{var saved=localStorage.getItem('rating_'+id);if(saved){{slider.value=saved;slider.nextElementSibling.textContent=saved;}}}}}}); 
updateProgress();
</script>
</body>
</html>"""


def main():
    print("=== Morning Brief Generator ===")
    now_israel = datetime.now(tz=ISRAEL_TZ)
    date_display = now_israel.strftime("%A, %B %d, %Y")
    update_time = now_israel.strftime("%H:%M")
    year = now_israel.year

    genai.configure(api_key=GEMINI_API_KEY)

    all_articles = {}
    for category, feeds in RSS_FEEDS.items():
        print("[" + category.upper() + "] Fetching articles...")
        articles = fetch_articles(category, feeds)
        print("  Found " + str(len(articles)) + " articles")
        all_articles[category] = articles

    enriched = {"geopolitics": [], "economy": [], "technology": []}
    article_counter = 0

    for category, articles in all_articles.items():
        print("[" + category.upper() + "] Running Gemini analysis...")
        for article in articles:
            article_id = "art_" + str(article_counter)
            article_counter += 1
            print("  Analyzing: " + article["title"][:60])
            ai_data = analyze_with_gemini(article)
            enriched[category].append({"article": article, "ai": ai_data, "id": article_id})
            time.sleep(1.5)

    col_left_html = '<div class="col-section-label">Geopolitics</div>'
    for item in enriched["geopolitics"]:
        col_left_html += build_article_html(item["article"], item["ai"], item["id"])

    col_center_html = ""
    tech_items = enriched["technology"]
    econ_items = enriched["economy"]
    if tech_items:
        col_center_html += build_article_html(tech_items[0]["article"], tech_items[0]["ai"], tech_items[0]["id"], is_main=True)
    if econ_items:
        col_center_html += build_article_html(econ_items[0]["article"], econ_items[0]["ai"], econ_items[0]["id"])

    col_right_html = '<div class="col-section-label">In Brief</div>'
    for item in tech_items[1:]:
        col_right_html += build_article_html(item["article"], item["ai"], item["id"])
    for item in econ_items[1:]:
        col_right_html += build_article_html(item["article"], item["ai"], item["id"])

    html = HTML_TEMPLATE.format(
        date_display=date_display,
        update_time=update_time,
        year=year,
        total_articles=article_counter,
        col_left=col_left_html,
        col_center=col_center_html,
        col_right=col_right_html,
    )

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "index.html"
    output_path.write_text(html, encoding="utf-8")

    print("Done. Generated " + str(article_counter) + " articles.")
    print("Saved to " + str(output_path))


if __name__ == "__main__":
    main()