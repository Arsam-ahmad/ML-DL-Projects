import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from SchoolProjects.DeepLearningGoogleDrive.guardian_fetcher import GuardianFetcher

# -----------------------------
# Config
# -----------------------------
SPLIT_NAME = "labelled_dev"          # FEVER v1.0 split with gold evidence ids
KEEP_LABELS = {"SUPPORTS", "REFUTES"}

# Evidence chunking: build a multi-sentence chunk around the gold sentence id
MAX_CHARS = 420                      # "squish sentences together" target size
EXPAND_ORDER = "lr"                  # expand left then right (or "rl")

# Embedding model
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cuda"

OUT_CLAIMS = "claims_with_label.npz"
OUT_EVID = "embedded_evidence.npz"
OUT_MAP = "claim_to_gold_evidence.npz"


# -----------------------------
# FEVER wiki "lines" parsing
# -----------------------------
def parse_fever_lines(lines_str: str) -> dict[int, str]:
    """
    FEVER wiki_pages 'lines' format looks like:
      0\tSentence text...\tEntity1\tEntity2...
      1\tSentence text...\tEntity...
    We want ONLY the sentence text column (parts[1]).
    Returns: {line_id: sentence_text}
    """
    sent_map = {}
    if not lines_str:
        return sent_map

    for raw in lines_str.split("\n"):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split("\t")
        if len(parts) < 2:
            continue

        # parts[0] = line id, parts[1] = sentence, parts[2:] = extra junk (entities/links)
        try:
            line_id = int(parts[0])
        except ValueError:
            continue

        sentence = parts[1].strip()
        if sentence:
            sent_map[line_id] = sentence

    return sent_map


def build_chunk(sent_map: dict[int, str], center_id: int, max_chars: int) -> str | None:
    """
    Start with the gold sentence. Expand to adjacent sentences until we hit max_chars.
    Returns a multi-sentence chunk.
    """
    if center_id not in sent_map:
        return None

    left = right = center_id
    chunk = sent_map[center_id]

    def try_add_left(cur: str, l: int) -> tuple[str, int, bool]:
        if (l - 1) in sent_map:
            cand = sent_map[l - 1] + " " + cur
            if len(cand) <= max_chars:
                return cand, l - 1, True
        return cur, l, False

    def try_add_right(cur: str, r: int) -> tuple[str, int, bool]:
        if (r + 1) in sent_map:
            cand = cur + " " + sent_map[r + 1]
            if len(cand) <= max_chars:
                return cand, r + 1, True
        return cur, r, False

    while True:
        grew = False

        if EXPAND_ORDER == "lr":
            chunk, left, did = try_add_left(chunk, left)
            grew |= did
            chunk, right, did = try_add_right(chunk, right)
            grew |= did
        else:
            chunk, right, did = try_add_right(chunk, right)
            grew |= did
            chunk, left, did = try_add_left(chunk, left)
            grew |= did

        if not grew:
            break

    return chunk


# -----------------------------
# Main pipeline
# -----------------------------
def main():
    # 1) Load FEVER claims (v1.0)
    fever = load_dataset("fever", "v1.0")
    labelled = fever[SPLIT_NAME]

    # Quick sanity: label distribution
    label_counts = {}
    for r in labelled:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
    print("Label counts in split:", label_counts)

    # 2) Build claim -> {label, evidence_pairs[(page, sent_id), ...]}
    claim_map = {}  # claim_text -> {"label": str, "evidence": set[(page, sent_id)]}
    bad_rows = 0

    for r in labelled:
        label = r.get("label", "")
        if label not in KEEP_LABELS:
            continue

        page = r.get("evidence_wiki_url", "")
        sent_id = r.get("evidence_sentence_id", -1)
        claim = r.get("claim", "")

        if not claim or not page or sent_id is None or sent_id < 0:
            bad_rows += 1
            continue

        entry = claim_map.setdefault(claim, {"label": label, "evidence": set()})
        # If a claim ever shows conflicting labels (rare), keep the first and skip conflicts
        if entry["label"] != label:
            continue

        entry["evidence"].add((page, int(sent_id)))

    claims = list(claim_map.keys())
    labels = [claim_map[c]["label"] for c in claims]

    # Hard stop if you hit the "Total claims: 0" situation again
    print("Total claims:", len(claims))
    print("Total supports:", sum(1 for y in labels if y == "SUPPORTS"))
    print("Total refutes:", sum(1 for y in labels if y == "REFUTES"))
    print("Bad/empty rows skipped:", bad_rows)
    if len(claims) == 0:
        raise RuntimeError(
            "No claims survived filtering. You are likely loading the wrong split/config "
            "or label strings don't match. The label_counts print above tells you what exists."
        )

    # 3) Gather all gold pages
    gold_pages = set()
    for info in claim_map.values():
        for (p, _) in info["evidence"]:
            gold_pages.add(p)

    print("Unique gold pages in RAG DB:", len(gold_pages))

    # 4) Stream FEVER wiki_pages to load ONLY needed pages
    wiki_stream = load_dataset("fever", "wiki_pages", split="wikipedia_pages", streaming=True)

    page_to_sentmap = {}
    found = 0
    pbar = tqdm(total=len(gold_pages), desc="Loading gold wiki pages (stream)")
    for row in wiki_stream:
        pid = row.get("id", "")
        if pid in gold_pages and pid not in page_to_sentmap:
            sent_map = parse_fever_lines(row.get("lines", ""))
            # fallback: if 'lines' missing, try splitting 'text' (rare)
            if not sent_map and row.get("text"):
                # treat whole text as one "sentence"
                sent_map = {0: row["text"].strip()}
            page_to_sentmap[pid] = sent_map
            found += 1
            pbar.update(1)
            if found >= len(gold_pages):
                break
    pbar.close()

    missing_pages = gold_pages - set(page_to_sentmap.keys())
    print("Missing gold pages (not found in wiki_pages stream):", len(missing_pages))

    # 5) Build evidence chunks around GOLD sentence ids (multi-sentence)
    evidence_texts = []
    evidence_index = {}  # (page, sent_id) -> chunk_idx
    for pid, sent_map in page_to_sentmap.items():
        for sid in list(sent_map.keys()):
            # NOTE: we only want chunks for sentence ids that are actually referenced by claims,
            # so we’ll fill this from claim_map next (more efficient).
            pass

    # Efficient: only create chunks for sentence_ids that appear in claim_map
    gold_pairs = set()
    for info in claim_map.values():
        gold_pairs |= info["evidence"]

    for (pid, sid) in tqdm(gold_pairs, desc="Building gold evidence chunks"):
        sent_map = page_to_sentmap.get(pid)
        if not sent_map:
            continue
        chunk = build_chunk(sent_map, sid, MAX_CHARS)
        if not chunk:
            continue

        # Prefixing page id can help a bit, but keep it short
        chunk_text = f"[{pid}] {chunk}"

        if (pid, sid) not in evidence_index:
            evidence_index[(pid, sid)] = len(evidence_texts)
            evidence_texts.append(chunk_text)

    print("Total evidence chunks:", len(evidence_texts))
    if len(evidence_texts) == 0:
        raise RuntimeError("No evidence chunks created. Likely sentence_id parsing mismatch.")

    # 6) Save claims + labels
    np.savez(
        OUT_CLAIMS,
        inputs=np.array(claims, dtype=object),
        labels=np.array(labels, dtype=object),
    )
    print(f"Saved {OUT_CLAIMS}")

    # 7) Save mapping: claim idx -> list of evidence chunk indices (gold)
    claim_gold = []
    for c in claims:
        idxs = []
        for (pid, sid) in claim_map[c]["evidence"]:
            j = evidence_index.get((pid, sid))
            if j is not None:
                idxs.append(j)
        claim_gold.append(np.array(sorted(set(idxs)), dtype=np.int32))

    np.savez(OUT_MAP, claim_gold=np.array(claim_gold, dtype=object))
    print(f"Saved {OUT_MAP}")

    evidence_texts += get_real_evidence_guardian()

    # 8) Embed evidence chunks
    embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)
    embeddings = embedder.encode(evidence_texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)

    np.savez(
        OUT_EVID,
        embeddings=np.array(embeddings),
        texts=np.array(evidence_texts, dtype=object),
    )
    print(f"Saved {OUT_EVID}")

def get_real_evidence_guardian(GUARDIAN_API_KEY="14eaf206-d783-4bee-b7f8-bcf9361aff06"):
    fetcher = GuardianFetcher(api_key=GUARDIAN_API_KEY)
    global_evidence_strings = fetcher.fetch_and_extract_claims(
        from_date="2025-01-01",
        to_date="2025-12-10",
        page_size=20,  # Start small for testing
        max_pages=1000  # Just 2 pages for testing
    )
    return global_evidence_strings


if __name__ == "__main__":
    main()
