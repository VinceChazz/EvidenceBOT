"""Sentence-aware text chunking.

Chunks never span two publications, always keep their parent document id and
metadata, and overlap slightly so a sentence is never cut away from its
supporting context.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

from processing.models import Document, Passage
from utils.helpers import clean_text, split_sentences
from utils.logging_config import get_logger

logger = get_logger("processing.chunking")


def chunk_text(
    text: str,
    chunk_size_words: int = 170,
    overlap_words: int = 40,
    min_chunk_words: int = 12,
) -> List[str]:
    """Split ``text`` into overlapping, sentence-aligned chunks."""

    text = clean_text(text)
    if not text:
        return []
    words = text.split()
    if len(words) <= chunk_size_words:
        return [text] if len(words) >= min(3, min_chunk_words) else [text]

    sentences = split_sentences(text) or [text]
    chunks: List[str] = []
    current: List[str] = []
    current_words = 0

    for sentence in sentences:
        sentence_words = sentence.split()
        # A single very long sentence is hard-split on word boundaries.
        if len(sentence_words) > chunk_size_words:
            if current:
                chunks.append(" ".join(current))
                current, current_words = [], 0
            for start in range(0, len(sentence_words), chunk_size_words):
                piece = " ".join(sentence_words[start : start + chunk_size_words])
                if len(piece.split()) >= min_chunk_words:
                    chunks.append(piece)
            continue

        if current_words + len(sentence_words) > chunk_size_words and current:
            chunk = " ".join(current).strip()
            if len(chunk.split()) >= min(3, min_chunk_words):
                chunks.append(chunk)
            # Build overlap from the tail of the finished chunk.
            overlap: List[str] = []
            overlap_count = 0
            for sent in reversed(current):
                overlap.insert(0, sent)
                overlap_count += len(sent.split())
                if overlap_count >= overlap_words:
                    break
            current = overlap
            current_words = overlap_count

        current.append(sentence)
        current_words += len(sentence_words)

    if current:
        chunk = " ".join(current).strip()
        if chunks and len(chunk.split()) < min_chunk_words:
            chunks[-1] = (chunks[-1] + " " + chunk).strip()
        elif len(chunk.split()) >= min_chunk_words or not chunks:
            chunks.append(chunk)

    return [c for c in chunks if c.strip()]


def chunk_documents(
    documents: Iterable[Document],
    chunk_size_words: int = 170,
    overlap_words: int = 40,
    min_chunk_words: int = 12,
) -> List[Passage]:
    """Chunk every document into :class:`Passage` objects with metadata."""

    passages: List[Passage] = []
    for document in documents:
        pieces = chunk_text(
            document.text_passage,
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
            min_chunk_words=min_chunk_words,
        )
        if not pieces:
            pieces = [document.title] if document.title else []
        for index, piece in enumerate(pieces):
            passages.append(Passage.from_document(document, piece, chunk_index=index))
    logger.debug("chunked documents into %d passages", len(passages))
    return passages


def chunk_passages_for_display(passage_texts: Sequence[str]) -> List[str]:
    """Utility used by the UI to shorten very long passages in expanders."""

    return [clean_text(t) for t in passage_texts]
