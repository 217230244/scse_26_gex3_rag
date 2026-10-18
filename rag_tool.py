"""Retrieve evidence from the single supplied policy file."""

from collections import Counter
from functools import lru_cache
import math
import re

from langchain_core.tools import tool

from policy_loader import load_policy_documents


TOKEN = re.compile(r"[a-z0-9]+")
PHONE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{6,}\d(?!\w)")
STOP_WORDS = {
    "a", "an", "and", "are", "can", "do", "for", "from", "how", "i",
    "in", "is", "it", "me", "my", "of", "on", "or", "should", "the",
    "to", "what", "when", "where", "with", "you",
}


def _tokens(text):
    return [word for word in TOKEN.findall(text.lower()) if word not in STOP_WORDS]


@lru_cache(maxsize=1)
def _index():
    # Keep policy_loader.py unchanged. It parses the one required policy file.
    documents = load_policy_documents()
    counts = [Counter(_tokens(doc.page_content)) for doc in documents]
    frequency = Counter()
    for terms in counts:
        frequency.update(terms.keys())
    average_length = sum(sum(terms.values()) for terms in counts) / len(counts)
    return documents, counts, frequency, average_length


def retrieve_policy_documents(question, top_k=5):
    """Rank policy sections with BM25 and return the most relevant ones."""
    documents, counts, frequency, average_length = _index()
    query = set(_tokens(question))
    if not query:
        return []

    ranked = []
    total = len(documents)
    k1, b = 1.5, 0.75
    for document, terms in zip(documents, counts):
        length = sum(terms.values())
        score = 0.0
        for word in query:
            count = terms[word]
            if not count:
                continue
            idf = math.log1p((total - frequency[word] + 0.5) / (frequency[word] + 0.5))
            score += idf * count * (k1 + 1) / (
                count + k1 * (1 - b + b * length / average_length)
            )
        if score:
            ranked.append((score, document))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [document for _, document in ranked[:top_k]]


def verified_service_desk_phone():
    """Return a number only if the supplied policies explicitly state one."""
    documents, _, _, _ = _index()
    for document in documents:
        for line in document.page_content.splitlines():
            if re.search(r"service desk", line, re.IGNORECASE) and re.search(
                r"phone|telephone|call", line, re.IGNORECASE
            ):
                match = PHONE.search(line)
                if match:
                    return match.group().strip()
    return None


@tool
def search_policy(question: str) -> str:
    """Search the supplied university IT policies for evidence relevant to a question."""
    documents = retrieve_policy_documents(question)
    if not documents:
        return "No relevant policy section found."
    return "\n\n".join(
        f"[policy_id={doc.metadata['policy_id']} topic={doc.metadata['topic']}]\n"
        f"{doc.page_content}"
        for doc in documents
    )
