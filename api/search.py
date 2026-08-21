"""
Vercel serverless function for internet search.
Uses WSGI-compatible handler.
"""
import json
import re
import time
import os

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

_ddgs = None


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
        results = []
        search_results = _ddgs.text(query, max_results=5)
        if hasattr(search_results, '__iter__'):
            for item in search_results:
                results.append({
                    "title": item.get("title", ""),
                    "body": item.get("body", ""),
                    "url": item.get("href", ""),
                })
        ms = (time.perf_counter() - t0) * 1000

        answer = _build_answer(query, results)

        response = {
            "answer": answer or "I couldn't find a good answer. Try rephrasing.",
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


def _build_answer(query, results):
    if not results:
        return None
    query_words = set(query.lower().split())
    stop = {'what','is','the','a','an','who','how','when','where','why','tell','me','about',
            'can','you','do','does','in','on','at','to','for','of','and','or','my','your','this','that','it'}
    meaningful = query_words - stop
    scored = []
    for r in results[:5]:
        body = r.get("body", "")
        title = r.get("title", "")
        if not body:
            continue
        text = (title + " " + body).lower()
        score = sum(1 for w in meaningful if w in text)
        if re.search(r'\b(is|are|was|were)\b', text):
            score += 2
        cleaned = re.sub(r'^\d+\s+(hours?|days?|weeks?|months?|minutes?)\s+ago\s*[-–—]\s*', '', body)
        cleaned = re.sub(r'^(Today|Yesterday)\s*[-–—]\s*', '', cleaned)
        cleaned = re.sub(r'^[A-Z][a-z]+\s+\d{1,2},\s*\d{4}\s*[–—-]\s*', '', cleaned)
        if cleaned.strip():
            scored.append((score, cleaned.strip()))
    scored.sort(key=lambda x: -x[0])
    parts = []
    seen = set()
    for score, text in scored[:3]:
        key = text[:80].lower()
        if key not in seen:
            seen.add(key)
            parts.append(text)
    return "\n\n".join(parts) if parts else None
