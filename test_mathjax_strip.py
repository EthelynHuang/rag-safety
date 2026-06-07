"""
Verify strip_mathjax_css on three known-contaminated articles.
Prints before/after character counts and 300-char windows around the seam.
Does NOT re-ingest anything.
"""
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from ingestion.fetch import strip_mathjax_css

CACHE_ROOT = Path.home() / ".cache/huggingface/hub/datasets--StampyAI--alignment-research-dataset/snapshots"
snapshots = sorted(CACHE_ROOT.iterdir())
if not snapshots:
    print("ERROR: no Stampy snapshot found in HF cache"); sys.exit(1)
SNAPSHOT = snapshots[-1]

TARGETS = [
    ("alignmentforum.jsonl", "The Inner Alignment Problem"),
    ("alignmentforum.jsonl", "Corrigibility as Constrained Optimisation"),
    ("lesswrong.jsonl",      "Inference from a Mathematical Description"),
]

SEP = "─" * 72

def find_record(jsonl_path: Path, title: str) -> dict | None:
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if title.lower() in (row.get("title") or "").lower():
                return row
    return None

for filename, title in TARGETS:
    path = SNAPSHOT / filename
    row = find_record(path, title)
    if row is None:
        print(f"NOT FOUND: '{title}' in {filename}"); continue

    raw = row.get("text", "")
    cleaned = strip_mathjax_css(raw)

    css_start = raw.find('.mjx-')
    last_ff   = raw.rfind('@font-face')
    css_end   = raw.find('}', last_ff) if last_ff != -1 else -1

    print(SEP)
    print(f"ARTICLE : {title}")
    print(f"SOURCE  : {filename}")
    print(f"BEFORE  : {len(raw):,} chars")
    print(f"AFTER   : {len(cleaned):,} chars  ({len(raw)-len(cleaned):,} chars removed)")
    print()

    if css_start == -1:
        print("  [no .mjx- found — article was already clean]")
    else:
        print(f"  CSS block in raw text: chars {css_start}–{css_end}")
        print()

        # Show 300 chars immediately before where the CSS started
        pre = raw[max(0, css_start - 300): css_start]
        print("  ── BEFORE (last 300 chars before CSS block) ──")
        print(repr(pre))
        print()

        # Show 300 chars immediately after where the CSS ended
        post_raw = raw[css_end + 1: css_end + 301]
        print("  ── AFTER CSS in raw (first 300 chars after closing brace) ──")
        print(repr(post_raw))
        print()

        # Show the seam in the cleaned text
        seam_pos = css_start  # in cleaned text, this is where the join happened
        seam = cleaned[max(0, seam_pos - 150): seam_pos + 150]
        print("  ── SEAM in cleaned text (±150 chars around join point) ──")
        print(repr(seam))
        print()

        # Sanity checks
        still_has_css = '.mjx-' in cleaned or '@font-face' in cleaned
        print(f"  CSS still present in cleaned text: {still_has_css}")
        if still_has_css:
            print("  WARNING: residual CSS detected — manual inspection needed")

print(SEP)
print("Done. No data was modified or re-ingested.")
