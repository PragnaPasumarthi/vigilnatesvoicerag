"""
Vercel serverless function for internet search.
Fetches full page content for complete answers.
"""
import json
import re
import time
import urllib.request
import urllib.error
from html.parser import HTMLParser

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

_ddgs = None


class TextExtractor(HTMLParser):
    """Extract readable text from HTML."""
    def __init__(self):
        super().__init__()
        self.result = []
        self.skip = False
        self.skip_tags = {'script', 'style', 'nav', 'header', 'footer', 'aside', 'noscript', 'iframe'}
        self.current_tag = ''

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        if tag in self.skip_tags:
            self.skip = True

    def handle_endtag(self, tag):
        if tag in self.skip_tags:
            self.skip = False

    def handle_data(self, data):
        if not self.skip:
            text = data.strip()
            if text and len(text) > 20:
                self.result.append(text)

    def get_text(self):
        return ' '.join(self.result[:50])  # First 50 meaningful chunks


def fetch_page_content(url, max_chars=3000):
    """Fetch and extract text content from a URL."""
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        with urllib.request.urlopen(req, timeout=5) as resp:
            html = resp.read().decode('utf-8', errors='ignore')

        extractor = TextExtractor()
        extractor.feed(html)
        text = extractor.get_text()

        # Clean up
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:max_chars]
    except Exception:
        return None


def app(environ, start_response):
    """WSGI application for Vercel."""
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
        body = json.dumps({"error": "Not found"}).encode()
        start_response("404 Not Found", list(headers.items()))
        return [body]

    try:
        content_length = int(environ.get("CONTENT_LENGTH", 0))
        request_body = environ["wsgi.input"].read(content_length)
        data = json.loads(request_body)
        query = data.get("query", "").strip()

        if not query:
            body = json.dumps({"answer": "Please ask a question."}).encode()
            start_response("200 OK", list(headers.items()))
            return [body]

        global _ddgs
        if _ddgs is None:
            _ddgs = DDGS()

        t0 = time.perf_counter()

        # Search DuckDuckGo
        results = []
        search_results = _ddgs.text(query, max_results=5)
        if hasattr(search_results, '__iter__'):
            for item in search_results:
                results.append({
                    "title": item.get("title", ""),
                    "body": item.get("body", ""),
                    "url": item.get("href", ""),
                })

        # Fetch full content from top result
        full_content = None
        if results and results[0].get("url"):
            full_content = fetch_page_content(results[0]["url"])

        ms = (time.perf_counter() - t0) * 1000

        # Build comprehensive answer
        answer = _build_answer(query, results, full_content)

        response = {
            "answer": answer or "I couldn't find a complete answer. Try rephrasing.",
            "sources": [{"title": r["title"], "url": r["url"], "body": r["body"][:200]} for r in results[:3]],
            "latency_ms": round(ms, 1),
        }
        body = json.dumps(response, ensure_ascii=False).encode()
        start_response("200 OK", list(headers.items()))
        return [body]

    except Exception as e:
        body = json.dumps({"error": str(e), "answer": f"Search error: {e}"}).encode()
        start_response("200 OK", list(headers.items()))
        return [body]


def _build_answer(query, results, full_content=None):
    """Build a complete answer from search results and full page content."""
    if not results:
        return None

    query_words = set(query.lower().split())
    stop = {'what','is','the','a','an','who','how','when','where','why','tell','me','about',
            'can','you','do','does','in','on','at','to','for','of','and','or','my','your','this','that','it'}
    meaningful = query_words - stop

    parts = []

    # 1. Use full page content if available (most complete)
    if full_content and len(full_content) > 200:
        # Extract relevant paragraphs
        paragraphs = full_content.split('.')
        relevant = []
        for p in paragraphs:
            p = p.strip()
            if len(p) > 30:
                # Check if paragraph is relevant to query
                p_lower = p.lower()
                if any(w in p_lower for w in meaningful):
                    relevant.append(p + '.')
                elif len(relevant) < 3:  # Include first few for context
                    relevant.append(p + '.')

        if relevant:
            parts.append('\n\n'.join(relevant[:8]))

    # 2. Add search snippets for additional context
    for r in results[:3]:
        body = r.get("body", "")
        title = r.get("title", "")
        if not body:
            continue
        # Clean date prefixes
        cleaned = re.sub(r'^\d+\s+(hours?|days?|weeks?|months?|minutes?)\s+ago\s*[-–—]\s*', '', body)
        cleaned = re.sub(r'^(Today|Yesterday)\s*[-–—]\s*', '', cleaned)
        cleaned = re.sub(r'^[A-Z][a-z]+\s+\d{1,2},\s*\d{4}\s*[–—-]\s*', '', cleaned)
        if cleaned.strip() and len(cleaned) > 30:
            # Check it's not already in parts
            if not any(cleaned[:50] in p for p in parts):
                parts.append(cleaned.strip())

    return '\n\n'.join(parts) if parts else None
