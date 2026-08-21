"""
Vercel serverless function for internet search.
"""
import json
import re
import time

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

_ddgs = None


def handler(request, response):
    """Vercel Python handler."""
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
        "Content-Type": "application/json; charset=utf-8",
    }

    if request.method == "OPTIONS":
        response.status_code = 204
        for k, v in headers.items():
            response.headers[k] = v
        return

    if request.method != "POST":
        response.status_code = 405
        response.headers.update(headers)
        response.body = json.dumps({"error": "Method not allowed"})
        return

    try:
        body = json.loads(request.body)
        query = body.get("query", "").strip()
        if not query:
            response.status_code = 200
            response.headers.update(headers)
            response.body = json.dumps({"error": "Empty query", "answer": "Please ask a question."})
            return

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

        response.status_code = 200
        response.headers.update(headers)
        response.body = json.dumps({
            "answer": answer or "I couldn't find a good answer. Try rephrasing.",
            "sources": [{"title": r["title"], "url": r["url"], "body": r["body"][:200]} for r in results[:3]],
            "latency_ms": round(ms, 1),
        }, ensure_ascii=False)

    except Exception as e:
        response.status_code = 200
        response.headers.update(headers)
        response.body = json.dumps({"error": str(e), "answer": f"Search error: {e}"})


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
