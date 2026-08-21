"""Vercel Python serverless function: internet search → concise ChatGPT-style answer"""
import json, re, time
from flask import Flask, request as flask_request, jsonify

try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ddgs", "-q"])
        from ddgs import DDGS

app = Flask(__name__)


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

    try:
        results = DDGS().text(query, region=lang_to_region(lang), max_results=8)
    except Exception:
        return jsonify({
            "answer": "I couldn't search the internet right now. Please try again later.",
            "sources": [],
            "searchTime": round((time.time() - t0) * 1000),
            "safe": True,
        })

    if not results:
        return jsonify({
            "answer": f"I couldn't find information about '{query}'.",
            "sources": [],
            "searchTime": round((time.time() - t0) * 1000),
            "safe": True,
        })

    sources = []
    all_text = ""
    for r in results[:6]:
        title = r.get("title", "")
        body_text = r.get("body", "")
        href = r.get("href", "")
        if title and body_text:
            sources.append({"title": title, "url": href, "snippet": body_text[:200]})
            all_text += f"{title}\n{body_text}\n\n"

    answer = build_concise_answer(query, all_text, results)
    search_time = round((time.time() - t0) * 1000)

    return jsonify({
        "answer": answer,
        "sources": sources,
        "searchTime": search_time,
        "safe": True,
    })


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
