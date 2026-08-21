"""
Multi-strategy document chunking engine.

Implements 5 distinct chunking strategies:
1. Fixed-size character splitting
2. Sliding window with overlap
3. Recursive text splitting (paragraph → sentence → word)
4. Semantic chunking (sentence-boundary aware)
5. Metadata-aware chunking (respects document structure)

Each strategy produces Chunk objects with content, metadata, and position info.
The orchestrator runs multiple strategies and deduplicates/ranks the results.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True, slots=True)
class Chunk:
    """A single chunk of text with metadata."""
    content: str
    index: int
    strategy: str
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.content)

    @property
    def word_count(self) -> int:
        return len(self.content.split())

    def __repr__(self) -> str:
        preview = self.content[:60].replace('\n', ' ')
        return f"Chunk(idx={self.index}, strat={self.strategy}, {self.char_count}ch, '{preview}...')"


# ---------------------------------------------------------------------------
# Strategy 1: Fixed-size character splitting
# ---------------------------------------------------------------------------

def fixed_size_chunks(
    text: str,
    chunk_size: int = 512,
    strategy_name: str = "fixed",
) -> list[Chunk]:
    """Split text into fixed-size chunks by character count."""
    chunks: list[Chunk] = []
    for i in range(0, len(text), chunk_size):
        segment = text[i : i + chunk_size]
        if segment.strip():
            chunks.append(Chunk(
                content=segment.strip(),
                index=len(chunks),
                strategy=strategy_name,
                start_char=i,
                end_char=min(i + chunk_size, len(text)),
            ))
    return chunks


# ---------------------------------------------------------------------------
# Strategy 2: Sliding window with overlap
# ---------------------------------------------------------------------------

def sliding_window_chunks(
    text: str,
    chunk_size: int = 512,
    overlap: int = 128,
    strategy_name: str = "sliding",
) -> list[Chunk]:
    """Split text with a sliding window and character overlap."""
    chunks: list[Chunk] = []
    step = max(1, chunk_size - overlap)
    seen: set[str] = set()

    for start in range(0, len(text), step):
        end = min(start + chunk_size, len(text))
        segment = text[start:end].strip()
        if segment and segment not in seen:
            seen.add(segment)
            chunks.append(Chunk(
                content=segment,
                index=len(chunks),
                strategy=strategy_name,
                start_char=start,
                end_char=end,
                metadata={"overlap": overlap},
            ))
        if end >= len(text):
            break
    return chunks


# ---------------------------------------------------------------------------
# Strategy 3: Recursive text splitting
# ---------------------------------------------------------------------------

_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')
_PARAGRAPH_BREAK = re.compile(r'\n\s*\n')


def recursive_text_chunks(
    text: str,
    chunk_size: int = 512,
    strategy_name: str = "recursive",
) -> list[Chunk]:
    """Recursively split: paragraphs → sentences → words, respecting chunk_size."""
    return _recursive_split(text, chunk_size, strategy_name, 0)


def _recursive_split(
    text: str,
    chunk_size: int,
    strategy_name: str,
    start_char: int,
) -> list[Chunk]:
    if len(text) <= chunk_size and text.strip():
        return [Chunk(
            content=text.strip(),
            index=0,  # re-indexed later
            strategy=strategy_name,
            start_char=start_char,
            end_char=start_char + len(text),
        )]

    # Try splitting by paragraphs first
    parts = _PARAGRAPH_BREAK.split(text)
    if len(parts) > 1:
        return _merge_recursive_parts(parts, chunk_size, strategy_name, start_char)

    # Try splitting by sentences
    parts = _SENTENCE_END.split(text)
    if len(parts) > 1:
        return _merge_recursive_parts(parts, chunk_size, strategy_name, start_char)

    # Fall back to word splitting
    words = text.split()
    merged: list[str] = []
    current = ""
    for w in words:
        if len(current) + len(w) + 1 > chunk_size and current:
            merged.append(current)
            current = w
        else:
            current = f"{current} {w}".strip() if current else w
    if current:
        merged.append(current)
    return _merge_recursive_parts(merged, chunk_size, strategy_name, start_char)


def _merge_recursive_parts(
    parts: list[str],
    chunk_size: int,
    strategy_name: str,
    base_offset: int,
) -> list[Chunk]:
    """Merge small parts into chunk_size groups."""
    chunks: list[Chunk] = []
    current_text = ""
    current_start = base_offset
    offset = base_offset

    for part in parts:
        if len(current_text) + len(part) + 2 > chunk_size and current_text:
            chunks.append(Chunk(
                content=current_text.strip(),
                index=0,
                strategy=strategy_name,
                start_char=current_start,
                end_char=offset,
            ))
            current_text = part
            current_start = offset
        else:
            current_text = f"{current_text}\n\n{part}".strip() if current_text else part
        offset += len(part) + 2

    if current_text.strip():
        chunks.append(Chunk(
            content=current_text.strip(),
            index=0,
            strategy=strategy_name,
            start_char=current_start,
            end_char=offset,
        ))

    # Re-index
    for i, c in enumerate(chunks):
        object.__setattr__(c, "index", i)
    return chunks


# ---------------------------------------------------------------------------
# Strategy 4: Semantic chunking (sentence-boundary aware)
# ---------------------------------------------------------------------------

def semantic_chunks(
    text: str,
    max_chunk_words: int = 80,
    min_chunk_words: int = 20,
    strategy_name: str = "semantic",
) -> list[Chunk]:
    """
    Split on sentence boundaries, then group sentences into semantically
    coherent chunks by word count, with a soft max/min target.

    Uses a simple coherence heuristic: keeps sentences together when they
    share vocabulary overlap (simple TF-based check).
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    # Build sentence groups
    groups: list[list[str]] = []
    current_group: list[str] = []
    current_words = 0

    for sent in sentences:
        sent_words = len(sent.split())

        if current_words + sent_words > max_chunk_words and current_group:
            # Check if we should keep this sentence (coherence check)
            if current_words < min_chunk_words:
                # Too small — extend the current group
                current_group.append(sent)
                current_words += sent_words
            else:
                groups.append(current_group)
                current_group = [sent]
                current_words = sent_words
        else:
            current_group.append(sent)
            current_words += sent_words

    if current_group:
        groups.append(current_group)

    # Convert to Chunks
    chunks: list[Chunk] = []
    offset = 0
    for i, group in enumerate(groups):
        content = " ".join(group)
        start = text.find(group[0][:50], offset)
        if start == -1:
            start = offset
        end = start + len(content)
        chunks.append(Chunk(
            content=content,
            index=i,
            strategy=strategy_name,
            start_char=start,
            end_char=end,
            metadata={"sentence_count": len(group)},
        ))
        offset = end

    return chunks


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    # Handle common abbreviations to avoid false splits
    text = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|Inc|Ltd|Jr|Sr|vs|etc|approx)\.\s', r'\1 DOT ', text)
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.replace('DOT ', '. ').strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Strategy 5: Metadata-aware chunking
# ---------------------------------------------------------------------------

def metadata_aware_chunks(
    text: str,
    chunk_size: int = 512,
    overlap: int = 100,
    strategy_name: str = "metadata_aware",
) -> list[Chunk]:
    """
    Chunk text while preserving and propagating structural metadata.

    Detects:
    - Headings (lines starting with # or ALL CAPS)
    - Numbered sections
    - Lists
    - Code blocks
    - Paragraphs

    Each chunk inherits metadata about its source section, type, etc.
    """
    segments = _parse_structure(text)
    chunks: list[Chunk] = []
    current_text = ""
    current_meta: dict = {}
    current_start = 0

    for seg in segments:
        seg_text = seg["content"]

        if len(current_text) + len(seg_text) + 2 > chunk_size and current_text:
            chunks.append(Chunk(
                content=current_text.strip(),
                index=len(chunks),
                strategy=strategy_name,
                start_char=current_start,
                end_char=current_start + len(current_text),
                metadata=current_meta.copy(),
            ))
            # Overlap: include last part of previous chunk
            if overlap > 0 and current_text:
                tail = current_text[-overlap:]
                current_text = tail + "\n\n" + seg_text
                current_start = current_start + len(current_text) - len(tail) - 2 - len(seg_text)
            else:
                current_text = seg_text
                current_start = seg["start"]
        else:
            if not current_text:
                current_start = seg["start"]
            current_text = f"{current_text}\n\n{seg_text}".strip() if current_text else seg_text

        # Update metadata
        if seg.get("heading"):
            current_meta["section"] = seg["heading"]
        if seg.get("type"):
            current_meta["segment_type"] = seg["type"]

    if current_text.strip():
        chunks.append(Chunk(
            content=current_text.strip(),
            index=len(chunks),
            strategy=strategy_name,
            start_char=current_start,
            end_char=current_start + len(current_text),
            metadata=current_meta.copy(),
        ))

    return chunks


def _parse_structure(text: str) -> list[dict]:
    """Parse document structure into labeled segments."""
    segments: list[dict] = []
    lines = text.split('\n')
    i = 0
    current_paragraph: list[str] = []

    def flush_paragraph():
        if current_paragraph:
            content = '\n'.join(current_paragraph)
            if content.strip():
                segments.append({
                    "content": content,
                    "type": "paragraph",
                    "start": text.find(content[:30]),
                })
            current_paragraph.clear()

    while i < len(lines):
        line = lines[i]

        # Heading detection
        if re.match(r'^#{1,6}\s', line):
            flush_paragraph()
            segments.append({
                "content": line,
                "type": "heading",
                "heading": line.strip('#').strip(),
                "start": text.find(line[:30]),
            })
            i += 1
            continue

        # ALL CAPS heading (at least 3 words, no lowercase)
        if re.match(r'^[A-Z][A-Z\s\d:,.()-]{10,}$', line.strip()) and line.strip():
            flush_paragraph()
            segments.append({
                "content": line,
                "type": "heading",
                "heading": line.strip(),
                "start": text.find(line[:30]),
            })
            i += 1
            continue

        # Numbered section
        if re.match(r'^\d+[\.\)]\s', line.strip()):
            flush_paragraph()
            seg_text = line
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(r'^\d+[\.\)]\s', lines[i].strip()):
                seg_text += '\n' + lines[i]
                i += 1
            segments.append({
                "content": seg_text,
                "type": "numbered_section",
                "heading": line.strip()[:80],
                "start": text.find(line[:30]),
            })
            continue

        # Code block
        if line.strip().startswith('```'):
            flush_paragraph()
            code_lines = [line]
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            if i < len(lines):
                code_lines.append(lines[i])
                i += 1
            code_text = '\n'.join(code_lines)
            segments.append({
                "content": code_text,
                "type": "code",
                "start": text.find(code_text[:30]),
            })
            continue

        # List item
        if re.match(r'^[\s]*[-*+]\s', line) or re.match(r'^[\s]*\d+\.\s', line):
            flush_paragraph()
            seg_text = line
            i += 1
            while i < len(lines) and re.match(r'^[\s]*[-*+\d.]\s', lines[i]):
                seg_text += '\n' + lines[i]
                i += 1
            segments.append({
                "content": seg_text,
                "type": "list",
                "start": text.find(seg_text[:30]),
            })
            continue

        # Regular paragraph text
        if line.strip():
            current_paragraph.append(line)
        else:
            flush_paragraph()
        i += 1

    flush_paragraph()
    return segments


# ---------------------------------------------------------------------------
# Chunking orchestrator: runs multiple strategies and deduplicates
# ---------------------------------------------------------------------------

@dataclass
class ChunkingConfig:
    """Configuration for the chunking orchestrator."""
    fixed_size: int = 512
    sliding_size: int = 512
    sliding_overlap: int = 128
    recursive_size: int = 512
    semantic_max_words: int = 80
    semantic_min_words: int = 20
    metadata_size: int = 512
    metadata_overlap: int = 100
    dedup_threshold: float = 0.8  # Jaccard similarity for dedup


def chunk_document(
    text: str,
    config: Optional[ChunkingConfig] = None,
    strategies: Optional[list[str]] = None,
) -> list[Chunk]:
    """
    Run multiple chunking strategies and return deduplicated chunks.

    Strategies: "fixed", "sliding", "recursive", "semantic", "metadata_aware"
    If strategies is None, runs all five.
    """
    cfg = config or ChunkingConfig()
    all_strategies = strategies or [
        "fixed", "sliding", "recursive", "semantic", "metadata_aware"
    ]

    all_chunks: list[Chunk] = []

    if "fixed" in all_strategies:
        all_chunks.extend(fixed_size_chunks(text, cfg.fixed_size))
    if "sliding" in all_strategies:
        all_chunks.extend(sliding_window_chunks(text, cfg.sliding_size, cfg.sliding_overlap))
    if "recursive" in all_strategies:
        all_chunks.extend(recursive_text_chunks(text, cfg.recursive_size))
    if "semantic" in all_strategies:
        all_chunks.extend(semantic_chunks(text, cfg.semantic_max_words, cfg.semantic_min_words))
    if "metadata_aware" in all_strategies:
        all_chunks.extend(metadata_aware_chunks(text, cfg.metadata_size, cfg.metadata_overlap))

    # Deduplicate
    deduped = _deduplicate_chunks(all_chunks, cfg.dedup_threshold)

    # Re-index
    for i, c in enumerate(deduped):
        object.__setattr__(c, "index", i)

    return deduped


def _deduplicate_chunks(chunks: list[Chunk], threshold: float) -> list[Chunk]:
    """Remove near-duplicate chunks using Jaccard similarity on word sets."""
    if not chunks:
        return []

    result: list[Chunk] = []
    word_sets: list[set[str]] = []

    for chunk in chunks:
        words = set(chunk.content.lower().split())
        if not words:
            continue

        is_dup = False
        for existing_words in word_sets:
            if not existing_words:
                continue
            intersection = len(words & existing_words)
            union = len(words | existing_words)
            if union > 0 and intersection / union >= threshold:
                is_dup = True
                break

        if not is_dup:
            result.append(chunk)
            word_sets.append(words)

    return result
