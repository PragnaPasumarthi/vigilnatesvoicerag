"""
Vercel serverless function for internet search.
Called by the chatbot when user asks a question.
"""
import json
import re
import time

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

_ddgs = None


def handler(request):
    """Handle Vercel serverless request."""
    # CORS headers
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }

    if request.method == "OPTIONS":
        return Response("", status=204, headers=headers)

    if request.method != "POST":
        return Response(json.dumps({"error": "Method not allowed"}), status=405, headers=headers)

    try:
        body = json.loads(request.body)
        query = body.get("query", "").strip()
        if not query:
            return Response(json.dumps({"error": "Empty query", "answer": "Please ask a question."}),
                          status=200, headers={**headers, "Content-Type": "application/json"})

        # Search the internet
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

        # Build answer
        answer = _build_answer(query, results)

        response_data = {
            "answer": answer or "I couldn't find a good answer. Try rephrasing your question.",
            "sources": [{"title": r["title"], "url": r["url"], "body": r["body"][:200]} for r in results[:3]],
            "latency_ms": round(ms, 1),
        }

        return Response(json.dumps(response_data, ensure_ascii=False),
                       status=200,
                       headers={**headers, "Content-Type": "application/json"})

    except Exception as e:
        return Response(json.dumps({"error": str(e), "answer": f"Search error: {e}"}),
                       status=200,
                       headers={**headers, "Content-Type": "application/json"})


def _build_answer(query, results):
    """Build a complete answer from search results."""
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
        # Clean date prefixes
        cleaned = re.sub(r'^\d+\s+(hours?|days?|weeks?|months?|minutes?)\s+ago\s*[-–—]\s*', '', body)
        cleaned = re.sub(r'^(Today|Yesterday)\s*[-–—]\s*', '', cleaned)
        cleaned = re.sub(r'^[A-Z][a-z]+\s+\d{1,2},\s*\d{4}\s*[–—-]\s*', '', cleaned)
        if cleaned.strip():
            scored.append((score, cleaned.strip()))

    scored.sort(key=lambda x: -x[0])

    # Combine top 3 results for complete answer
    parts = []
    seen = set()
    for score, text in scored[:3]:
        key = text[:80].lower()
        if key not in seen:
            seen.add(key)
            parts.append(text)

    return "\n\n".join(parts) if parts else None


class Response:
    def __init__(self, body, status=200, headers=None):
        self.body = body
        self.status_code = status
        self.headers = headers or {}
