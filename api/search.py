"""Vercel serverless function: internet search → concise ChatGPT-style answer"""
import json, re, sys, os, time

try:
    from duckduckgo_search import DDGS
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "duckduckgo-search", "-q"])
    from duckduckgo_search import DDGS


def handler(request):
    """Handle POST requests with JSON body containing 'query'."""
    if request.method != "OPTIONS":
        content_type = request.headers.get("content-type", "")
        if "json" not in content_type:
            return response(400, {"error": "Content-Type must be application/json"})

    if request.method == "OPTIONS":
        return response(200, {}, headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        })

    try:
        body = request.json()
        query = body.get("query", "").strip()
        lang = body.get("lang", "en")
    except Exception:
        return response(400, {"error": "Invalid JSON body"})

    if not query:
        return response(400, {"error": "Query is required"})

    t0 = time.time()

    # Search the internet
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, region=lang_to_region(lang), max_results=8))
    except Exception as e:
        return response(200, {
            "answer": f"I couldn't search the internet right now. Please try again later.",
            "sources": [],
            "searchTime": round((time.time() - t0) * 1000),
            "safe": True,
        })

    if not results:
        return response(200, {
            "answer": f"I couldn't find information about '{query}' on the internet.",
            "sources": [],
            "searchTime": round((time.time() - t0) * 1000),
            "safe": True,
        })

    # Build sources
    sources = []
    all_text = ""
    for r in results[:6]:
        title = r.get("title", "")
        body = r.get("body", "")
        href = r.get("href", "")
        if title and body:
            sources.append({"title": title, "url": href, "snippet": body[:200]})
            all_text += f"{title}\n{body}\n\n"

    # Build concise answer from raw text
    answer = build_concise_answer(query, all_text, results)

    search_time = round((time.time() - t0) * 1000)

    return response(200, {
        "answer": answer,
        "sources": sources,
        "searchTime": search_time,
        "safe": True,
    })


def build_concise_answer(query, raw_text, results):
    """Build a clean, concise answer from search results — ChatGPT style."""
    
    # Extract all meaningful sentences
    all_sentences = extract_clean_sentences(raw_text)
    
    if not all_sentences:
        # Fallback: just use the first result body
        if results:
            return clean_text(results[0].get("body", "No answer found."))
        return "No answer found."
    
    # Find the most relevant sentences to the query
    query_words = set(re.findall(r'\w{3,}', query.lower()))
    stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'way', 'who', 'did', 'get', 'let', 'say', 'she', 'too', 'use', 'this', 'that', 'with', 'have', 'from', 'they', 'been', 'said', 'each', 'make', 'like', 'than', 'them', 'then', 'what', 'when', 'your', 'will', 'there', 'their', 'about', 'would', 'could', 'other', 'which', 'after', 'these', 'some', 'only', 'also', 'into', 'very', 'just', 'being', 'first', 'where', 'while'}
    
    scored = []
    for s in all_sentences:
        words = set(re.findall(r'\w{3,}', s.lower()))
        relevant = words & query_words - stop_words
        if relevant:
            # Prefer shorter, more direct sentences
            score = len(relevant) * 2 - (len(s) > 150) * 1
            scored.append((score, len(s), s))
    
    scored.sort(key=lambda x: (-x[0], x[1]))
    
    # Take top 3-5 sentences
    selected = []
    seen = set()
    for score, length, sentence in scored[:5]:
        key = sentence[:50].lower()
        if key not in seen:
            seen.add(key)
            selected.append(sentence)
    
    if not selected:
        # Fallback: use first result body, cleaned up
        if results:
            body = results[0].get("body", "")
            return clean_text(body)
        return "No answer found."
    
    # Join into a clean answer
    answer = " ".join(selected)
    
    # Clean up formatting
    answer = clean_text(answer)
    
    # Limit length — keep it concise
    if len(answer) > 800:
        # Cut at sentence boundary
        cut = answer[:800].rfind('.')
        if cut > 400:
            answer = answer[:cut + 1]
        else:
            answer = answer[:800].strip() + "..."
    
    return answer


def extract_clean_sentences(text):
    """Extract clean, meaningful sentences from raw text."""
    # Remove common scraped noise
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
        r'(?i)menu',
        r'(?i)search\s+(?:this|the)',
    ]
    
    for p in noise:
        text = re.sub(p, '', text)
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    clean = []
    for s in sentences:
        s = s.strip()
        # Skip short/empty
        if len(s) < 20:
            continue
        # Skip sentences that are mostly non-alphabetic
        alpha = sum(1 for c in s if c.isalpha())
        if alpha < len(s) * 0.5:
            continue
        # Skip navigation-like text
        if re.match(r'^(?:home|about|contact|menu|search|login|sign|close|open|back|next|prev|toggle|click)', s, re.I):
            continue
        # Skip text with too many pipe characters (nav menus)
        if s.count('|') > 2:
            continue
        # Clean up
        s = re.sub(r'\s+', ' ', s).strip()
        if s:
            clean.append(s)
    
    return clean


def clean_text(text):
    """Clean up raw scraped text."""
    # Remove weird characters
    text = re.sub(r'[^\w\s.,;:!?\-\'\"()/—–&%$#@+=\[\]{}]', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Fix multiple punctuation
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


def response(status, body, headers=None):
    """Build a Vercel serverless response."""
    hdrs = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }
    if headers:
        hdrs.update(headers)
    
    return {
        "statusCode": status,
        "headers": hdrs,
        "body": json.dumps(body),
    }
