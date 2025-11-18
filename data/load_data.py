# search_action_openalex_env.py
import os
import requests
from typing import List, Optional

# Get URL from environment variable (default to OpenAlex)
OPENALEX_BASE = os.getenv("OPENALEX_BASE", "https://api.openalex.org/works")
OPENALEX_MAILTO = os.getenv("OPENALEX_MAILTO", "your_email@example.com")


def search_paper(search_query: str, per_page: int = 5) -> List[dict]:
    """
    Search OpenAlex and return a list of papers.

    Args:
        search_query: Search query string
        per_page: Number of results to return (default: 5)

    Returns:
        List of dictionaries with paper information:
        - title: str
        - abstract: str
        - citations: int
        - references: int
    """
    try:
        params = {
            "search": search_query,
            "per_page": per_page,
            "sort": "relevance_score:desc",
            "mailto": OPENALEX_MAILTO
        }
        resp = requests.get(OPENALEX_BASE, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        works = data.get("results", [])

        extracted = []
        for w in works:
            title = w.get("title") or ""
            # Reconstruct abstract from inverted index if available
            abstract_inv = w.get("abstract_inverted_index")
            if abstract_inv:
                maxpos = max(p for pos in abstract_inv.values() for p in pos)
                arr = [""] * (maxpos + 1)
                for token, positions in abstract_inv.items():
                    for p in positions:
                        arr[p] = token
                abstract = " ".join(w for w in arr if w)
            else:
                abstract = ""

            citation_count = w.get("cited_by_count", 0)
            reference_count = len(w.get("referenced_works") or [])

            extracted.append({
                "title": title,
                "abstract": abstract,
                "citations": citation_count,
                "references": reference_count
            })
        return extracted
    except Exception as e:
        print(f"Error searching papers: {e}")
        return []
