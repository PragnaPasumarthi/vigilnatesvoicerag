"""Vercel Python serverless function: internet search → concise ChatGPT-style answer"""
import json, re, time, urllib.parse
from flask import Flask, request as flask_request, jsonify

try:
    import requests
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests

app = Flask(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


@app.route("/api/search", methods=["POST", "OPTIONS"])
def search():
    if flask_request.method == "OPTIONS":
        return "", 200, {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }

    try:
        body = flask_request.get_json(force=True)
        query = body.get("query", "").strip()
        lang = body.get("lang", "en")
    except Exception:
        return jsonify({"error": "Invalid JSON body"}), 400

    if not query:
        return jsonify({"error": "Query is required"}), 400

    t0 = time.time()

    # Try multiple search approaches
    results = []
    
    # Approach 1: DuckDuckGo HTML search
    try:
        results = search_ddg_html(query, lang)
    except Exception:
        pass
    
    # Approach 2: DuckDuckGo instant answer API
    if not results:
        try:
            results = search_ddg_instant(query)
        except Exception:
            pass
    
    # Approach 3: Wikipedia API
    if not results:
        try:
            results = search_wikipedia(query, lang)
        except Exception:
            pass

    if not results:
        return jsonify({
            "answer": f"I couldn't find information about '{query}' at the moment.",
            "sources": [],
            "searchTime": round((time.time() - t0) * 1000),
            "safe": True,
        })

    # Build sources
    sources = []
    all_text = ""
    for r in results[:6]:
        title = r.get("title", "")
        body_text = r.get("body", "")
        url = r.get("url", "")
        if title and body_text:
            sources.append({"title": title, "url": url, "snippet": body_text[:200]})
            all_text += f"{title}\n{body_text}\n\n"

    answer = build_concise_answer(query, all_text, results)
    search_time = round((time.time() - t0) * 1000)

    return jsonify({
        "answer": answer,
        "sources": sources,
        "searchTime": search_time,
        "safe": True,
    })


def search_ddg_html(query, lang="en"):
    """Search DuckDuckGo via HTML scraping."""
    url = "https://html.duckduckgo.com/html/"
    params = {"q": query, "kl": lang_to_region(lang)}
    
    resp = requests.post(url, data=params, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    
    results = []
    # Extract results from HTML
    links = re.findall(r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>', resp.text, re.DOTALL)
    snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</(?:a|span|div)', resp.text, re.DOTALL)
    
    for i, (href, title) in enumerate(links[:8]):
        # Decode URL
        if 'uddg=' in href:
            href = urllib.parse.unquote(href.split('uddg=')[1].split('&')[0])
        
        title = re.sub(r'<[^>]+>', '', title).strip()
        snippet = re.sub(r'<[^>]+>', '', snippets[i]).strip() if i < len(snippets) else ""
        
        if title and snippet:
            results.append({"title": title, "body": snippet, "url": href})
    
    return results


def search_ddg_instant(query):
    """Search DuckDuckGo instant answer API."""
    url = "https://api.duckduckgo.com/"
    params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
    
    resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    
    results = []
    
    # Abstract (main answer)
    abstract = data.get("Abstract", "")
    if abstract:
        results.append({
            "title": data.get("Heading", query),
            "body": abstract,
            "url": data.get("AbstractURL", ""),
        })
    
    # Related topics
    for topic in data.get("RelatedTopics", [])[:5]:
        if isinstance(topic, dict) and "Text" in topic:
            results.append({
                "title": topic.get("Text", "")[:60],
                "body": topic.get("Text", ""),
                "url": topic.get("FirstURL", ""),
            })
    
    return results


def search_wikipedia(query, lang="en"):
    """Search Wikipedia API."""
    lang_map = {"hi": "hi", "es": "es", "fr": "fr", "de": "de", "ja": "ja", "ar": "ar", "te": "te", "ta": "ta", "bn": "bn", "ko": "ko"}
    wiki_lang = lang_map.get(lang, "en")
    
    # Search for articles
    search_url = f"https://{wiki_lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": 5,
    }
    
    resp = requests.get(search_url, params=params, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    
    results = []
    for item in data.get("query", {}).get("search", []):
        title = item.get("title", "")
        snippet = re.sub(r'<[^>]+>', '', item.get("snippet", "")).strip()
        url = f"https://{wiki_lang}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
        
        if title and snippet:
            results.append({"title": title, "body": snippet, "url": url})
    
    return results


def build_concise_answer(query, raw_text, results):
    """Build a clean, concise answer from search results — ChatGPT style."""
    all_sentences = extract_clean_sentences(raw_text)

    if not all_sentences:
        if results:
            return clean_text(results[0].get("body", "No answer found."))
        return "No answer found."

    query_words = set(re.findall(r'\w{3,}', query.lower()))
    stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'way', 'who', 'did', 'get', 'let', 'say', 'she', 'too', 'use', 'this', 'that', 'with', 'have', 'from', 'they', 'been', 'said', 'each', 'make', 'like', 'than', 'them', 'then', 'what', 'when', 'your', 'will', 'there', 'their', 'about', 'would', 'could', 'other', 'which', 'after', 'these', 'some', 'only', 'also', 'into', 'very', 'just', 'being', 'first', 'where', 'while'}

    scored = []
    for s in all_sentences:
        words = set(re.findall(r'\w{3,}', s.lower()))
        relevant = words & query_words - stop_words
        if relevant:
            score = len(relevant) * 2 - (len(s) > 150) * 1
            scored.append((score, len(s), s))

    scored.sort(key=lambda x: (-x[0], x[1]))

    selected = []
    seen = set()
    for score, length, sentence in scored[:5]:
        key = sentence[:50].lower()
        if key not in seen:
            seen.add(key)
            selected.append(sentence)

    if not selected:
        if results:
            return clean_text(results[0].get("body", ""))
        return "No answer found."

    answer = " ".join(selected)
    answer = clean_text(answer)

    if len(answer) > 800:
        cut = answer[:800].rfind('.')
        if cut > 400:
            answer = answer[:cut + 1]
        else:
            answer = answer[:800].strip() + "..."

    return answer


def extract_clean_sentences(text):
    """Extract clean, meaningful sentences from raw text."""
    noise = [
        r'(?i)skip to (?:main|content|navigation)',
        r'(?i)click here to',
        r'(?i)subscribe to our',
        r'(?i)sign up for',
        r'(?i)cookie[s]?\s+(?:policy|notice|settings)',
        r'(?i)privacy\s+policy',
        r'(?i)terms\s+(?:of|&)\s+(?:service|use)',
        r'(?i)all\s+rights\s+reserved',
        r'(?i)copyright\s+\d{4}',
        r'(?i)loading\.\.\.',
        r'(?i)advertisement',
        r'(?i)read more',
        r'(?i)share this',
        r'(?i)follow us',
        r'(?i)like\s+this',
        r'(?i)comment\s+below',
        r'(?i)javascript is required',
        r'(?i)enable javascript',
        r'(?i)\d+\s*(?:views?|likes?|shares?)',
    ]

    for p in noise:
        text = re.sub(p, '', text)

    sentences = re.split(r'(?<=[.!?])\s+', text)

    clean = []
    for s in sentences:
        s = s.strip()
        if len(s) < 20:
            continue
        alpha = sum(1 for c in s if c.isalpha())
        if alpha < len(s) * 0.5:
            continue
        if re.match(r'^(?:home|about|contact|menu|search|login|sign|close|open|back|next|prev|toggle|click)', s, re.I):
            continue
        if s.count('|') > 2:
            continue
        s = re.sub(r'\s+', ' ', s).strip()
        if s:
            clean.append(s)

    return clean


def clean_text(text):
    """Clean up raw scraped text."""
    text = re.sub(r'[^\w\s.,;:!?\-\'\"()/—–&%$#@+=\[\]{}]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'([.!?])\1+', r'\1', text)
    return text


def lang_to_region(lang):
    """Map language code to DuckDuckGo region."""
    regions = {
        "en": "wt-wt", "hi": "in-en", "es": "es-es",
        "fr": "fr-fr", "de": "de-de", "ja": "jp-jp",
        "ar": "ar-eg", "te": "in-en", "ta": "in-en",
        "bn": "in-en", "ko": "kr-kr",
    }
    return regions.get(lang, "wt-wt")


# WSGI entry point for Vercel
application = app
