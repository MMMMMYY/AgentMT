"""Exact, resumable surface-style frequency extraction over all WildChat rows.

This module deliberately performs no style classification.  It streams the
dataset, deduplicates exact user turns, and counts only prompt/sentence boundary
phrases.  Raw user turns are never written to disk.  Chunk files contain only
hashes and aggregate phrase counts and make a long full-corpus run resumable.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import heapq
import json
import os
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from privacy_mt.prompt_style_collection import WILDCHAT_DATASET, normalize_text


DATASET_REVISION = "main"
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+|[\r\n]+")
# CJK ideographs are individual tokens; other Unicode letter sequences are
# retained as words.  This avoids the previous English-only extraction while
# keeping the tokenizer dependency-free.
TOKEN_RE = re.compile(
    r"<url>|<email>|<number>|"
    r"[\u3400-\u4dbf\u4e00-\u9fff]|"
    r"[\u3040-\u30ffー]+|[\uac00-\ud7af]+|"
    r"[^\W\d_]+(?:['’][^\W\d_]+)?",
    re.UNICODE | re.IGNORECASE,
)


def tokenize_multilingual(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(normalize_text(text))]


def sentence_segments(text: str, max_sentences: int = 12) -> list[str]:
    normalized = normalize_text(text)
    return [part.strip() for part in SENTENCE_SPLIT_RE.split(normalized) if part.strip()][
        :max_sentences
    ]


def boundary_phrases(
    content: str,
    language: str,
    start_max_n: int = 8,
    end_max_n: int = 6,
) -> set[tuple[str, str, int, str]]:
    """Return each boundary phrase at most once for one user turn."""

    found: set[tuple[str, str, int, str]] = set()

    def add(tokens: list[str], start_position: str, end_position: str) -> None:
        if not tokens:
            return
        for n in range(1, min(start_max_n, len(tokens)) + 1):
            found.add((language, start_position, n, " ".join(tokens[:n])))
        for n in range(1, min(end_max_n, len(tokens)) + 1):
            found.add((language, end_position, n, " ".join(tokens[-n:])))

    add(tokenize_multilingual(content), "prompt_start", "prompt_end")
    for segment in sentence_segments(content):
        add(tokenize_multilingual(segment), "sentence_start", "sentence_end")
    return found


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_chunk(
    work_dir: Path,
    chunk_id: int,
    counts: Counter[tuple[str, str, int, str]],
    hashes: set[bytes],
) -> tuple[Path, Path]:
    counts_path = work_dir / f"counts-{chunk_id:05d}.jsonl.gz"
    hashes_path = work_dir / f"hashes-{chunk_id:05d}.txt.gz"
    counts_tmp = counts_path.with_suffix(counts_path.suffix + ".tmp")
    hashes_tmp = hashes_path.with_suffix(hashes_path.suffix + ".tmp")
    with gzip.open(counts_tmp, "wt", encoding="utf-8") as handle:
        for key in sorted(counts):
            handle.write(json.dumps([*key, counts[key]], ensure_ascii=False) + "\n")
    with gzip.open(hashes_tmp, "wt", encoding="ascii") as handle:
        for digest in sorted(hashes):
            handle.write(digest.hex() + "\n")
    os.replace(counts_tmp, counts_path)
    os.replace(hashes_tmp, hashes_path)
    return counts_path, hashes_path


def _load_seen_hashes(work_dir: Path, complete_chunks: int) -> set[bytes]:
    seen: set[bytes] = set()
    for chunk_id in range(complete_chunks):
        path = work_dir / f"hashes-{chunk_id:05d}.txt.gz"
        with gzip.open(path, "rt", encoding="ascii") as handle:
            seen.update(bytes.fromhex(line.strip()) for line in handle if line.strip())
    return seen


def _chunk_records(path: Path) -> Iterator[tuple[tuple[str, str, int, str], int]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            language, position, n_tokens, phrase, count = json.loads(line)
            yield (language, position, int(n_tokens), phrase), int(count)


def merge_chunks(
    work_dir: Path,
    complete_chunks: int,
    out_dir: Path,
    min_count: int,
    top_n: int,
) -> int:
    """Merge sorted chunk aggregates exactly and emit frequency-sorted files."""

    database = work_dir / "merged_counts.sqlite3"
    if database.exists():
        database.unlink()
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE styles ("
        "language TEXT, position TEXT, n_tokens INTEGER, phrase TEXT, "
        "turn_count INTEGER)"
    )
    paths = [work_dir / f"counts-{chunk_id:05d}.jsonl.gz" for chunk_id in range(complete_chunks)]
    streams = [_chunk_records(path) for path in paths]
    merged = heapq.merge(*streams, key=lambda item: item[0])
    retained = 0
    pending: list[tuple[str, str, int, str, int]] = []
    current_key: tuple[str, str, int, str] | None = None
    current_count = 0

    def keep(key: tuple[str, str, int, str] | None, count: int) -> None:
        nonlocal retained
        if key is None or count < min_count:
            return
        pending.append((*key, count))
        retained += 1
        if len(pending) >= 10_000:
            connection.executemany("INSERT INTO styles VALUES (?, ?, ?, ?, ?)", pending)
            connection.commit()
            pending.clear()

    for key, count in merged:
        if key != current_key:
            keep(current_key, current_count)
            current_key, current_count = key, count
        else:
            current_count += count
    keep(current_key, current_count)
    if pending:
        connection.executemany("INSERT INTO styles VALUES (?, ?, ?, ?, ?)", pending)
        connection.commit()
    connection.execute("CREATE INDEX styles_frequency ON styles(turn_count DESC)")
    connection.commit()

    fields = ["frequency_rank", "language", "position", "n_tokens", "phrase", "turn_count"]
    full_path = out_dir / "surface_style_frequencies.csv.gz"
    top_path = out_dir / "top_surface_styles.csv"
    query = (
        "SELECT language, position, n_tokens, phrase, turn_count FROM styles "
        "ORDER BY turn_count DESC, language, position, n_tokens, phrase"
    )
    with gzip.open(full_path, "wt", encoding="utf-8", newline="") as full_handle, top_path.open(
        "w", encoding="utf-8", newline=""
    ) as top_handle:
        full_writer = csv.writer(full_handle)
        top_writer = csv.writer(top_handle)
        full_writer.writerow(fields)
        top_writer.writerow(fields)
        for rank, row in enumerate(connection.execute(query), start=1):
            output_row = [rank, *row]
            full_writer.writerow(output_row)
            if rank <= top_n:
                top_writer.writerow(output_row)
    connection.close()
    return retained


def extract_full_corpus(
    out_dir: Path,
    dataset_name: str = WILDCHAT_DATASET,
    revision: str = DATASET_REVISION,
    min_count: int = 5,
    chunk_rows: int = 10_000,
    top_n: int = 50_000,
    max_rows: int | None = None,
) -> dict:
    from datasets import load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "_work"
    work_dir.mkdir(exist_ok=True)
    checkpoint_path = work_dir / "checkpoint.json"
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("dataset", dataset_name) != dataset_name:
            raise ValueError("Checkpoint dataset does not match --dataset")
        if checkpoint.get("revision", revision) != revision:
            raise ValueError("Checkpoint revision does not match --revision")
    else:
        checkpoint = {
            "next_row": 0,
            "complete_chunks": 0,
            "summary": {},
            "language_turn_counts": {},
        }

    complete_chunks = int(checkpoint["complete_chunks"])
    next_row = int(checkpoint["next_row"])
    summary = Counter(checkpoint.get("summary", {}))
    language_counts = Counter(checkpoint.get("language_turn_counts", {}))
    seen_hashes = _load_seen_hashes(work_dir, complete_chunks)
    print(
        json.dumps(
            {
                "status": "starting",
                "resume_at_row": next_row,
                "complete_chunks": complete_chunks,
                "seen_unique_turns": len(seen_hashes),
            }
        ),
        flush=True,
    )

    dataset = load_dataset(
        dataset_name,
        split="train",
        streaming=True,
        revision=revision,
    )
    if next_row:
        dataset = dataset.skip(next_row)

    chunk_counts: Counter[tuple[str, str, int, str]] = Counter()
    chunk_hashes: set[bytes] = set()
    rows_in_chunk = 0
    last_row = next_row - 1

    def flush(next_dataset_row: int) -> None:
        nonlocal complete_chunks, chunk_counts, chunk_hashes, rows_in_chunk
        if rows_in_chunk == 0:
            return
        _write_chunk(work_dir, complete_chunks, chunk_counts, chunk_hashes)
        seen_hashes.update(chunk_hashes)
        complete_chunks += 1
        state = {
            "next_row": next_dataset_row,
            "complete_chunks": complete_chunks,
            "summary": dict(summary),
            "language_turn_counts": dict(language_counts),
            "dataset": dataset_name,
            "revision": revision,
        }
        _atomic_json(checkpoint_path, state)
        print(
            json.dumps(
                {
                    "status": "checkpoint",
                    "next_row": next_dataset_row,
                    "complete_chunks": complete_chunks,
                    "unique_user_turns": summary["unique_user_turns"],
                    "chunk_unique_phrases": len(chunk_counts),
                }
            ),
            flush=True,
        )
        chunk_counts = Counter()
        chunk_hashes = set()
        rows_in_chunk = 0

    for local_index, row in enumerate(dataset):
        row_index = next_row + local_index
        if max_rows is not None and row_index >= max_rows:
            break
        last_row = row_index
        summary["conversations_seen"] += 1
        rows_in_chunk += 1
        for turn in row.get("conversation") or []:
            if turn.get("role") != "user":
                continue
            summary["all_user_turns"] += 1
            language = str(turn.get("language") or row.get("language") or "Unknown")
            language_counts[language] += 1
            content = str(turn.get("content") or "")
            normalized = normalize_text(content).casefold().strip()
            if not normalized:
                summary["empty_user_turns"] += 1
                continue
            digest = hashlib.sha256(normalized.encode("utf-8")).digest()
            if digest in seen_hashes or digest in chunk_hashes:
                summary["duplicate_user_turns_skipped"] += 1
                continue
            chunk_hashes.add(digest)
            summary["unique_user_turns"] += 1
            phrases = boundary_phrases(content, language)
            if not phrases:
                summary["untokenized_user_turns"] += 1
                continue
            summary["tokenized_user_turns"] += 1
            chunk_counts.update(phrases)
        if rows_in_chunk >= chunk_rows:
            flush(row_index + 1)

    flush(last_row + 1)
    retained = merge_chunks(work_dir, complete_chunks, out_dir, min_count, top_n)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_name,
        "revision": revision,
        "mode": "full streaming extraction" if max_rows is None else "bounded test extraction",
        "classification_performed": False,
        "raw_user_text_persisted": False,
        "exact_duplicate_turns_removed": True,
        "minimum_turn_count": min_count,
        "boundary_specification": {
            "positions": ["prompt_start", "prompt_end", "sentence_start", "sentence_end"],
            "maximum_start_tokens": 8,
            "maximum_end_tokens": 6,
            "maximum_sentences_per_turn": 12,
        },
        "summary": dict(summary),
        "language_turn_counts": language_counts.most_common(),
        "retained_surface_styles": retained,
        "complete_chunks": complete_chunks,
        "next_row": last_row + 1,
    }
    _atomic_json(out_dir / "extraction_report.json", report)
    (out_dir / "README.md").write_text(
        """# Full WildChat surface-style frequencies

This directory contains an exact, frequency-sorted inventory of recurring
prompt and sentence boundary phrases extracted from every processed WildChat
user turn. No style labels or seeded request-form filters were used. Counts are
separated by the dataset's language label, boundary position, and phrase
length. Exact duplicate user turns were counted once, and raw user messages
were never persisted. Only phrases meeting the report's minimum turn count are
included in the final inventory.

`surface_style_frequencies.csv.gz` is the complete frequency-sorted output.
`top_surface_styles.csv` is a convenient uncompressed preview. `_work/` holds
resumable aggregate chunks and message hashes, never raw conversations.
""",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", default=WILDCHAT_DATASET)
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--chunk-rows", type=int, default=10_000)
    parser.add_argument("--top-n", type=int, default=50_000)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args(argv)
    extract_full_corpus(
        out_dir=args.out,
        dataset_name=args.dataset,
        revision=args.revision,
        min_count=args.min_count,
        chunk_rows=args.chunk_rows,
        top_n=args.top_n,
        max_rows=args.max_rows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
