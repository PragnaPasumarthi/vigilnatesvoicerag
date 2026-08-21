"""
Vercel serverless function - smart concise answers.
"""
import json
import re
import time
import urllib.request
from html.parser import HTMLParser

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

_ddgs = None


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.result = []
        self.skip = False
        self.skip_tags = {'script', 'style', 'nav', 'header', 'footer', 'aside', 'noscript', 'iframe'}
    def handle_starttag(self, tag, attrs):
        if tag in self.skip_tags: self.skip = True
    def handle_endtag(self, tag):
        if tag in self.skip_tags: self.skip = False
    def handle_data(self, data):
        if not self.skip:
            text = data.strip()
            if text and len(text) > 15: self.result.append(text)
    def get_text(self):
        return ' '.join(self.result[:40])


def fetch_page_content(url, max_chars=2000):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=4) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
        ext = TextExtractor()
        ext.feed(html)
        text = re.sub(r'\s+', ' ', ext.get_text()).strip()
        return text[:max_chars]
    except:
        return None


def app(environ, start_response):
    method = environ.get("REQUEST_METHOD", "GET")
    path = environ.get("PATH_INFO", "/")
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
        "Content-Type": "application/json; charset=utf-8",
    }
    if method == "OPTIONS":
        start_response("204 No Content", list(headers.items()))
        return [b""]
    if method != "POST" or path != "/api/search":
        start_response("404 Not Found", list(headers.items()))
        return [json.dumps({"error": "Not found"}).encode()]

    try:
        length = int(environ.get("CONTENT_LENGTH", 0))
        data = json.loads(environ["wsgi.input"].read(length))
        query = data.get("query", "").strip()
        if not query:
            start_response("200 OK", list(headers.items()))
            return [json.dumps({"answer": "Please ask a question."}).encode()]

        global _ddgs
        if _ddgs is None: _ddgs = DDGS()

        t0 = time.perf_counter()

        # Search
        results = []
        for item in _ddgs.text(query, max_results=5):
            results.append({"title": item.get("title",""), "body": item.get("body",""), "url": item.get("href","")})

        # Score by relevance
        q_words = set(query.lower().split())
        stop = {'what','is','the','a','an','who','how','when','where','why','tell','me','about',
                'can','you','do','does','in','on','at','to','for','of','and','or','my','your','this','that','it'}
        meaningful = q_words - stop

        for r in results:
            title_l = r["title"].lower()
            body_l = r["body"].lower()
            r["_score"] = sum(2 for w in meaningful if w in title_l) + sum(1 for w in meaningful if w in body_l)
            if query.lower() in title_l: r["_score"] += 10

        results.sort(key=lambda x: -x.get("_score",0))

        # Fetch best source
        full_content = None
        for r in results:
            if r.get("url") and r.get("_score",0) > 0:
                full_content = fetch_page_content(r["url"])
                if full_content and len(full_content) > 100: break

        ms = (time.perf_counter() - t0) * 1000

        # Build SMART answer
        answer = _smart_answer(query, results, full_content, meaningful)

        start_response("200 OK", list(headers.items()))
        return [json.dumps({
            "answer": answer,
            "sources": [{"title": r["title"], "url": r["url"]} for r in results[:3]],
            "latency_ms": round(ms, 1),
        }, ensure_ascii=False).encode()]

    except Exception as e:
        start_response("200 OK", list(headers.items()))
        return [json.dumps({"error": str(e), "answer": f"Error: {e}"}).encode()]


def _smart_answer(query, results, full_content, meaningful):
    """Build a ChatGPT-style concise answer."""

    # 1. Extract the most relevant sentence(s) from full content
    if full_content:
        sentences = re.split(r'(?<=[.!?])\s+', full_content)
        relevant = []
        for s in sentences:
            s = s.strip()
            if len(s) < 20: continue
            s_lower = s.lower()
            # Score this sentence
            score = sum(1 for w in meaningful if w in s_lower)
            # Boost for definitions, key facts
            if re.search(r'\b(is|are|was|were|founded|established|located|capital)\b', s_lower):
                score += 2
            # Boost for sentences containing the query
            if any(w in s_lower for w in meaningful):
                score += 1
            if score > 0:
                relevant.append((score, s))

        relevant.sort(key=lambda x: -x[0])

        # Take top 2-3 most relevant sentences
        if relevant:
            parts = []
            seen = set()
            for score, s in relevant[:3]:
                # Clean
                s = re.sub(r'^\[\d+\]', '', s).strip()
                if s and s[:50] not in seen:
                    seen.add(s[:50])
                    parts.append(s)
            if parts:
                return ' '.join(parts)

    # 2. Fallback: use best search snippet (just the body, cleaned)
    for r in results[:2]:
        body = r.get("body", "")
        if not body: continue
        # Clean date prefix
        cleaned = re.sub(r'^\d+\s+(hours?|days?|weeks?|months?)\s+ago\s*[-–—]\s*', '', body)
        cleaned = re.sub(r'^(Today|Yesterday)\s*[-–—]\s*', '', cleaned)
        cleaned = re.sub(r'^[A-Z][a-z]+\s+\d{1,2},\s*\d{4}\s*[–—-]\s*', '', cleaned)
        if cleaned.strip() and len(cleaned) > 30:
            # Take just the first 1-2 sentences
            sents = re.split(r'(?<=[.!?])\s+', cleaned)
            return ' '.join(sents[:2])

    return "I couldn't find a concise answer. Try rephrasing your question."
