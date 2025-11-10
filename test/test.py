# extract_openalex_fields.py
import os, requests, json
from typing import Optional, List

OPENALEX_BASE = os.getenv("OPENALEX_BASE", "https://api.openalex.org/works")
OPENALEX_MAILTO = os.getenv("OPENALEX_MAILTO", "your_email@example.com")

def reconstruct_abstract(inverted_idx: Optional[dict]) -> str:
    """Ghép lại abstract từ abstract_inverted_index."""
    if not inverted_idx:
        return ""
    maxpos = max(p for pos in inverted_idx.values() for p in pos)
    arr = [""] * (maxpos + 1)
    for token, positions in inverted_idx.items():
        for p in positions:
            arr[p] = token
    return " ".join(w for w in arr if w)

def extract_title_abstract_citation(search_query: str, per_page: int = 5) -> List[dict]:
    """Gọi OpenAlex và chỉ trích ra title, abstract, citations."""
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
        abstract = reconstruct_abstract(w.get("abstract_inverted_index")) or ""
        citation_count = w.get("cited_by_count", 0)       
        reference_count = len(w.get("referenced_works") or [])
        extracted.append({
            "title": title,
            "abstract": abstract,
            "citations": citation_count,
            "references": reference_count
        })
    return extracted

if __name__ == "__main__":
    query = "Tree-Wasserstein optimal transport generative modeling"
    papers = extract_title_abstract_citation(query, per_page=5)
    print(f"Found {len(papers)} papers.\n")
    for i, p in enumerate(papers, 1):
        print(f"{i}. {p['title']}")
        print(f"   Citations: {p['citations']} | References: {p['references']}")
        print(f"   Abstract: {p['abstract'][:200]}...\n")
