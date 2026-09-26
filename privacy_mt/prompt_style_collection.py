"""Collect corpus-grounded request formats for prompt-style transformations.

The task benchmarks remain separate.  This module uses WildChat only to count
how real users formulate requests, and the Stanford Politeness Corpus to attach
politeness evidence.  It intentionally does not persist WildChat message text.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import requests


STACK_URL = (
    "https://zissou.infosci.cornell.edu/convokit/datasets/"
    "stack-exchange-politeness-corpus/stack-exchange-politeness-corpus.zip"
)
WIKI_URL = (
    "https://zissou.infosci.cornell.edu/convokit/datasets/"
    "wikipedia-politeness-corpus/wikipedia-politeness-corpus.zip"
)
WILDCHAT_ROWS_URL = "https://datasets-server.huggingface.co/rows"
WILDCHAT_DATASET = "allenai/WildChat-1M"

LABEL_NAMES = {-1: "impolite", 0: "neutral", 1: "polite"}

LEAD = (
    r"(?:^|[.!?]\s+)(?:(?:hi|hello|hey)(?:\s+there)?[!,.:\-\s]+)?"
    r"(?:(?:chatgpt|gpt|claude|assistant)[!,.:\-\s]+)?"
)


@dataclass(frozen=True)
class RequestFormat:
    format_id: str
    template: str
    family: str
    regex: re.Pattern[str]
    authorization_change_risk: str = "low"


def _rx(body: str) -> re.Pattern[str]:
    return re.compile(LEAD + body, re.IGNORECASE)


# Specific forms precede general modal and imperative forms.
REQUEST_FORMATS = (
    RequestFormat(
        "wondering_if_could",
        "I was wondering if you could {TASK}.",
        "mitigated_modal",
        _rx(r"i\s+(?:was|am|'m)\s+wondering\s+if\s+you\s+could\b"),
    ),
    RequestFormat(
        "appreciate_if_could",
        "I would appreciate it if you could {TASK}.",
        "gratitude_modal",
        _rx(r"i(?:\s+would|'d)\s+appreciate\s+it\s+if\s+you\s+could\b"),
    ),
    RequestFormat(
        "would_you_mind",
        "Would you mind {TASK}?",
        "mitigated_modal",
        _rx(r"would\s+you\s+mind\b"),
    ),
    RequestFormat(
        "do_you_mind",
        "Do you mind {TASK}?",
        "mitigated_modal",
        _rx(r"do\s+you\s+mind\b"),
    ),
    RequestFormat(
        "is_it_possible",
        "Is it possible for you to {TASK}?",
        "mitigated_modal",
        _rx(r"is\s+it\s+possible(?:\s+for\s+you)?\s+to\b"),
    ),
    RequestFormat(
        "are_you_able_to",
        "Are you able to {TASK}?",
        "modal_request",
        _rx(r"are\s+you\s+able\s+to\b"),
    ),
    RequestFormat(
        "it_would_be_great_if",
        "It would be great if you could {TASK}.",
        "mitigated_modal",
        _rx(r"it\s+would\s+be\s+(?:great|helpful|nice)\s+if\s+you\s+could\b"),
    ),
    RequestFormat(
        "id_be_grateful_if",
        "I'd be grateful if you could {TASK}.",
        "gratitude_modal",
        _rx(r"i(?:\s+would|'d)\s+be\s+grateful\s+if\s+you\s+could\b"),
    ),
    RequestFormat(
        "could_you_please",
        "Could you please {TASK}?",
        "polite_modal",
        _rx(r"could\s+you\s+please\b"),
    ),
    RequestFormat(
        "can_you_please",
        "Can you please {TASK}?",
        "polite_modal",
        _rx(r"can\s+you\s+please\b"),
    ),
    RequestFormat(
        "would_you_please",
        "Would you please {TASK}?",
        "polite_modal",
        _rx(r"would\s+you\s+please\b"),
    ),
    RequestFormat(
        "please_could_you",
        "Please, could you {TASK}?",
        "polite_modal",
        _rx(r"please[,]?\s+could\s+you\b"),
    ),
    RequestFormat(
        "please_can_you",
        "Please, can you {TASK}?",
        "polite_modal",
        _rx(r"please[,]?\s+can\s+you\b"),
    ),
    RequestFormat(
        "could_you_help_me",
        "Could you help me {TASK}?",
        "help_seeking",
        _rx(r"could\s+you\s+(?:please\s+)?help\s+me\b"),
    ),
    RequestFormat(
        "can_you_help_me",
        "Can you help me {TASK}?",
        "help_seeking",
        _rx(r"can\s+you\s+(?:please\s+)?help\s+me\b"),
    ),
    RequestFormat(
        "would_you_help_me",
        "Would you help me {TASK}?",
        "help_seeking",
        _rx(r"would\s+you\s+(?:please\s+)?help\s+me\b"),
    ),
    RequestFormat(
        "please_help_me",
        "Please help me {TASK}.",
        "help_seeking",
        _rx(r"please\s+help\s+me\b"),
    ),
    RequestFormat(
        "kindly",
        "Kindly {TASK}.",
        "polite_imperative",
        _rx(r"kindly\b"),
    ),
    RequestFormat(
        "please_imperative",
        "Please {TASK}.",
        "polite_imperative",
        _rx(r"please\b"),
    ),
    RequestFormat(
        "id_like_you_to",
        "I'd like you to {TASK}.",
        "desire_statement",
        _rx(r"i(?:\s+would|'d)\s+like\s+you\s+to\b"),
    ),
    RequestFormat(
        "i_need_you_to",
        "I need you to {TASK}.",
        "need_statement",
        _rx(r"i\s+need\s+you\s+to\b"),
    ),
    RequestFormat(
        "i_want_you_to",
        "I want you to {TASK}.",
        "want_statement",
        _rx(r"i\s+want\s+you\s+to\b"),
    ),
    RequestFormat(
        "could_you",
        "Could you {TASK}?",
        "modal_request",
        _rx(r"could\s+you\b"),
    ),
    RequestFormat(
        "can_you",
        "Can you {TASK}?",
        "modal_request",
        _rx(r"can\s+you\b"),
    ),
    RequestFormat(
        "would_you",
        "Would you {TASK}?",
        "modal_request",
        _rx(r"would\s+you\b"),
    ),
    RequestFormat(
        "help_me",
        "Help me {TASK}.",
        "help_seeking",
        _rx(r"help\s+me\b"),
    ),
    RequestFormat(
        "why_cant_you",
        "Why can't you {TASK}?",
        "challenging_request",
        _rx(r"why\s+can(?:no|')t\s+you\b"),
        "medium",
    ),
    RequestFormat(
        "just_imperative",
        "Just {TASK}.",
        "blunt_imperative",
        _rx(r"just\s+(?:find|book|send|check|search|schedule|buy|order|pay|upload|download|open|create|write|edit|fix|update|delete|summarize|analyse|analyze|tell|explain|generate|make|show|give)\b"),
        "medium",
    ),
    RequestFormat(
        "direct_imperative",
        "{TASK}.",
        "direct_imperative",
        _rx(r"(?:find|book|send|check|search|schedule|buy|order|pay|upload|download|open|create|write|edit|fix|update|delete|summarize|analyse|analyze|tell|explain|generate|make|show|give)\b"),
    ),
)

SUFFIX_PLEASE_RE = re.compile(r"(?:,|\s)\s*please[.!?\s]*$", re.IGNORECASE)
SUFFIX_THANKS_RE = re.compile(
    r"(?:,|\s)\s*(?:thanks|thank\s+you)(?:\s+(?:in\s+advance|so\s+much))?[.!?\s]*$",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    text = re.sub(r"```.*?```", " <CODE> ", text, flags=re.DOTALL)
    text = re.sub(r"https?://\S+|www\.\S+|<url>", " <URL> ", text, flags=re.I)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", " <EMAIL> ", text)
    text = re.sub(r"\b\d{5,}\b", " <NUMBER> ", text)
    return re.sub(r"\s+", " ", text).strip()


def detect_request_format(text: str) -> tuple[RequestFormat | None, tuple[str, ...]]:
    normalized = normalize_text(text)
    modifiers: list[str] = []
    if re.match(r"^\s*(?:hi|hello|hey)\b", normalized, re.I):
        modifiers.append("greeting_prefix")
    if SUFFIX_PLEASE_RE.search(normalized):
        modifiers.append("please_suffix")
    if SUFFIX_THANKS_RE.search(normalized):
        modifiers.append("thanks_suffix")
    for request_format in REQUEST_FORMATS:
        if request_format.regex.search(normalized):
            if request_format.format_id == "direct_imperative" and "please_suffix" in modifiers:
                return RequestFormat(
                    "please_suffix",
                    "{TASK}, please.",
                    "polite_imperative",
                    re.compile("$^"),
                ), tuple(modifiers)
            if request_format.format_id == "direct_imperative" and "thanks_suffix" in modifiers:
                return RequestFormat(
                    "thanks_suffix",
                    "{TASK}, thanks.",
                    "gratitude_imperative",
                    re.compile("$^"),
                ), tuple(modifiers)
            return request_format, tuple(modifiers)
    if "please_suffix" in modifiers:
        return RequestFormat(
            "please_suffix",
            "{TASK}, please.",
            "polite_imperative",
            re.compile("$^"),
        ), tuple(modifiers)
    if "thanks_suffix" in modifiers:
        return RequestFormat(
            "thanks_suffix",
            "{TASK}, thanks.",
            "gratitude_imperative",
            re.compile("$^"),
        ), tuple(modifiers)
    return None, tuple(modifiers)


TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|<url>|<email>|<number>")


def text_features(text: str) -> Counter[str]:
    tokens = TOKEN_RE.findall(normalize_text(text).lower())
    feats: Counter[str] = Counter("w=" + token for token in tokens)
    feats.update("b=" + a + "_" + b for a, b in zip(tokens, tokens[1:]))
    return feats


class MultinomialNB:
    """Small dependency-free classifier used only for corpus triage."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.class_docs: Counter[int] = Counter()
        self.class_totals: Counter[int] = Counter()
        self.feature_counts: dict[int, Counter[str]] = defaultdict(Counter)
        self.vocab: set[str] = set()

    def fit(self, rows: Iterable[dict]) -> "MultinomialNB":
        for row in rows:
            label = int(row["label"])
            feats = text_features(row["text"])
            self.class_docs[label] += 1
            self.feature_counts[label].update(feats)
            self.class_totals[label] += sum(feats.values())
            self.vocab.update(feats)
        return self

    def predict(self, text: str) -> int:
        feats = text_features(text)
        total_docs = sum(self.class_docs.values())
        vocab_size = max(1, len(self.vocab))
        scores: dict[int, float] = {}
        for label in sorted(self.class_docs):
            score = math.log(self.class_docs[label] / total_docs)
            denom = self.class_totals[label] + self.alpha * vocab_size
            counts = self.feature_counts[label]
            for feat, freq in feats.items():
                score += freq * math.log((counts[feat] + self.alpha) / denom)
            scores[label] = score
        return max(scores, key=scores.get)


def classification_metrics(gold: list[int], pred: list[int]) -> dict:
    labels = (-1, 0, 1)
    confusion = {str(g): {str(p): 0 for p in labels} for g in labels}
    for g, p in zip(gold, pred):
        confusion[str(g)][str(p)] += 1
    f1s = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "n": len(gold),
        "accuracy": sum(g == p for g, p in zip(gold, pred)) / len(gold),
        "macro_f1": sum(f1s) / len(f1s),
        "confusion": confusion,
    }


def download_file(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def read_stanford_zip(path: Path, source: str) -> list[dict]:
    with zipfile.ZipFile(path) as archive:
        member = next(name for name in archive.namelist() if name.endswith("utterances.jsonl"))
        with archive.open(member) as handle:
            rows = []
            for raw_line in handle:
                item = json.loads(raw_line)
                rows.append(
                    {
                        "source": source,
                        "source_id": item["id"],
                        "text": item["text"],
                        "score": float(item["meta"]["Normalized Score"]),
                        "label": int(item["meta"]["Binary"]),
                    }
                )
            return rows


def stratified_split(rows: list[dict], seed: int = 20260905) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    by_label: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[int(row["label"])].append(row)
    train, test = [], []
    for label_rows in by_label.values():
        rng.shuffle(label_rows)
        cut = int(len(label_rows) * 0.8)
        train.extend(label_rows[:cut])
        test.extend(label_rows[cut:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def evaluate_classifier(stanford_rows: list[dict]) -> tuple[MultinomialNB, dict]:
    train, test = stratified_split(list(stanford_rows))
    model = MultinomialNB().fit(train)
    report = {
        "random_stratified_holdout": classification_metrics(
            [int(row["label"]) for row in test],
            [model.predict(row["text"]) for row in test],
        )
    }
    for train_source, test_source in (("wikipedia", "stack_exchange"), ("stack_exchange", "wikipedia")):
        source_train = [row for row in stanford_rows if row["source"] == train_source]
        source_test = [row for row in stanford_rows if row["source"] == test_source]
        source_model = MultinomialNB().fit(source_train)
        report[f"train_{train_source}_test_{test_source}"] = classification_metrics(
            [int(row["label"]) for row in source_test],
            [source_model.predict(row["text"]) for row in source_test],
        )
    return MultinomialNB().fit(stanford_rows), report


def fetch_wildchat_page(offset: int, length: int = 100, retries: int = 8) -> dict:
    params = {
        "dataset": WILDCHAT_DATASET,
        "config": "default",
        "split": "train",
        "offset": offset,
        "length": length,
    }
    for attempt in range(retries):
        try:
            response = requests.get(WILDCHAT_ROWS_URL, params=params, timeout=120)
            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", 0) or 0)
                time.sleep(max(retry_after, min(60.0, 5.0 * (attempt + 1))))
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            if attempt + 1 == retries:
                raise
            time.sleep(min(60.0, 2**attempt))
    raise AssertionError("unreachable")


def sample_offsets(total_rows: int, pages: int, page_size: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    max_offset = max(0, total_rows - page_size)
    buckets = pages
    width = max(1, (max_offset + 1) // buckets)
    return sorted(min(max_offset, i * width + rng.randrange(width)) for i in range(buckets))


def collect_wildchat(
    model: MultinomialNB,
    pages: int,
    workers: int,
    seed: int,
) -> tuple[dict, dict[str, dict], list[int]]:
    total_rows = 837_989
    page_size = 100
    offsets = sample_offsets(total_rows, pages, page_size, seed)
    payloads: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_wildchat_page, offset, page_size): offset for offset in offsets}
        for future in as_completed(futures):
            offset = futures[future]
            payloads[offset] = future.result()

    summary = Counter()
    forms: dict[str, dict] = {}
    for request_format in REQUEST_FORMATS:
        forms[request_format.format_id] = {
            "format": request_format,
            "wildchat_count": 0,
            "predicted_labels": Counter(),
            "modifiers": Counter(),
            "word_counts": [],
        }
    forms["please_suffix"] = {
        "format": RequestFormat(
            "please_suffix", "{TASK}, please.", "polite_imperative", re.compile("$^")
        ),
        "wildchat_count": 0,
        "predicted_labels": Counter(),
        "modifiers": Counter(),
        "word_counts": [],
    }
    forms["thanks_suffix"] = {
        "format": RequestFormat(
            "thanks_suffix", "{TASK}, thanks.", "gratitude_imperative", re.compile("$^")
        ),
        "wildchat_count": 0,
        "predicted_labels": Counter(),
        "modifiers": Counter(),
        "word_counts": [],
    }

    for offset in offsets:
        for wrapped in payloads[offset].get("rows", []):
            summary["conversations_seen"] += 1
            row = wrapped.get("row", {})
            conversation = row.get("conversation") or []
            first_user = next((turn for turn in conversation if turn.get("role") == "user"), None)
            if not first_user:
                continue
            summary["first_user_turns"] += 1
            if first_user.get("language") != "English":
                continue
            summary["english_first_user_turns"] += 1
            if first_user.get("toxic"):
                summary["toxic_skipped"] += 1
                continue
            text = normalize_text(first_user.get("content") or "")
            words = text.split()
            if not 3 <= len(words) <= 120:
                summary["length_skipped"] += 1
                continue
            request_format, modifiers = detect_request_format(text)
            if request_format is None:
                summary["unmatched_english_turns"] += 1
                continue
            summary["matched_request_turns"] += 1
            bucket = forms[request_format.format_id]
            bucket["wildchat_count"] += 1
            bucket["predicted_labels"][model.predict(text)] += 1
            bucket["modifiers"].update(modifiers)
            bucket["word_counts"].append(len(words))
    return dict(summary), forms, offsets


def aggregate_stanford(rows: list[dict]) -> dict[str, dict]:
    result: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "scores": [], "labels": Counter(), "modifiers": Counter()}
    )
    for row in rows:
        request_format, modifiers = detect_request_format(row["text"])
        if request_format is None:
            continue
        bucket = result[request_format.format_id]
        bucket["count"] += 1
        bucket["scores"].append(float(row["score"]))
        bucket["labels"][int(row["label"])] += 1
        bucket["modifiers"].update(modifiers)
    return result


def evidence_class(stanford: dict, wildchat: dict) -> tuple[str, str]:
    if stanford.get("count", 0) >= 10:
        mean_score = statistics.mean(stanford["scores"])
        if mean_score >= 0.15:
            return "polite", "stanford_pattern_mean"
        if mean_score <= -0.15:
            return "impolite", "stanford_pattern_mean"
        return "neutral", "stanford_pattern_mean"
    predicted = wildchat.get("predicted_labels", Counter())
    if predicted:
        label = max((-1, 0, 1), key=lambda item: (predicted[item], item))
        return LABEL_NAMES[label], "stanford_trained_nb_majority"
    return "unclassified", "insufficient_evidence"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_outputs(
    out_dir: Path,
    stanford_rows: list[dict],
    classifier_report: dict,
    wildchat_summary: dict,
    wildchat_forms: dict[str, dict],
    offsets: list[int],
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stanford_agg = aggregate_stanford(stanford_rows)
    library = []
    for format_id, wildchat in wildchat_forms.items():
        request_format: RequestFormat = wildchat["format"]
        stanford = stanford_agg.get(format_id, {})
        label, label_basis = evidence_class(stanford, wildchat)
        scores = stanford.get("scores", [])
        word_counts = wildchat.get("word_counts", [])
        library.append(
            {
                "format_id": format_id,
                "template": request_format.template,
                "family": request_format.family,
                "assigned_class": label,
                "class_basis": label_basis,
                "authorization_change_risk": request_format.authorization_change_risk,
                "wildchat_count": int(wildchat["wildchat_count"]),
                "wildchat_predicted_polite": int(wildchat["predicted_labels"][1]),
                "wildchat_predicted_neutral": int(wildchat["predicted_labels"][0]),
                "wildchat_predicted_impolite": int(wildchat["predicted_labels"][-1]),
                "wildchat_median_words": statistics.median(word_counts) if word_counts else None,
                "stanford_count": int(stanford.get("count", 0)),
                "stanford_mean_score": statistics.mean(scores) if scores else None,
                "stanford_polite": int(stanford.get("labels", Counter())[1]),
                "stanford_neutral": int(stanford.get("labels", Counter())[0]),
                "stanford_impolite": int(stanford.get("labels", Counter())[-1]),
            }
        )
    library.sort(key=lambda item: (-item["wildchat_count"], item["format_id"]))

    write_json(out_dir / "format_library.json", library)
    write_json(out_dir / "classifier_evaluation.json", classifier_report)
    write_json(
        out_dir / "collection_report.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "purpose": "Collect expression formats; benchmark tasks come from separate task datasets.",
            "wildchat": {
                "dataset": WILDCHAT_DATASET,
                "sampling": "deterministic stratified offsets via public rows API",
                "sampled_offsets": offsets,
                **wildchat_summary,
                "raw_text_persisted": False,
            },
            "stanford": {
                "stack_exchange_rows": sum(row["source"] == "stack_exchange" for row in stanford_rows),
                "wikipedia_rows": sum(row["source"] == "wikipedia" for row in stanford_rows),
                "total_rows": len(stanford_rows),
            },
            "limitations": [
                "WildChat is a chatbot corpus, so request forms are filtered but not guaranteed to be executable agent tasks.",
                "Stanford labels come from human-human forum requests and require human validation after transfer to agent prompts.",
                "The Naive Bayes classifier is a triage model, not the final paper oracle.",
                "Format-level labels can be affected by the task inserted into a template.",
            ],
        },
    )

    with (out_dir / "format_library.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(library[0]))
        writer.writeheader()
        writer.writerows(library)

    with (out_dir / "stanford_labeled_requests.jsonl").open("w", encoding="utf-8") as handle:
        for row in stanford_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    readme = """# Prompt-style format collection v1

This directory separates **expression evidence** from the project task sources.
AI Agent Permissions, AgentCIBench, and AgentDojo remain the task sources.

- `format_library.csv` / `.json`: corpus-grounded request templates and evidence.
- `collection_report.json`: sampling details and limitations.
- `classifier_evaluation.json`: diagnostic evaluation of the Stanford-trained
  triage classifier.
- `stanford_labeled_requests.jsonl`: compact CC BY 4.0 export of the labeled
  Stack Exchange and Wikipedia requests used here.

WildChat text is processed in memory and is deliberately not persisted. Only
aggregate counts are written. Stanford Politeness Corpus data are from ConvoKit
and remain subject to CC BY 4.0 attribution. WildChat is ODC-BY.

The assigned format classes are candidate labels for manual audit. They are not
yet final experimental transformations, and medium-risk formats must not be
used until authorization-equivalence review is complete.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    return library


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/prompt-style-corpora"))
    parser.add_argument("--wildchat-pages", type=int, default=50)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args(argv)

    stack_zip = args.cache_dir / "stack-exchange-politeness-corpus.zip"
    wiki_zip = args.cache_dir / "wikipedia-politeness-corpus.zip"
    download_file(STACK_URL, stack_zip)
    download_file(WIKI_URL, wiki_zip)
    stanford_rows = read_stanford_zip(stack_zip, "stack_exchange")
    stanford_rows += read_stanford_zip(wiki_zip, "wikipedia")
    model, classifier_report = evaluate_classifier(stanford_rows)
    wildchat_summary, wildchat_forms, offsets = collect_wildchat(
        model=model,
        pages=args.wildchat_pages,
        workers=args.workers,
        seed=args.seed,
    )
    library = build_outputs(
        args.out,
        stanford_rows,
        classifier_report,
        wildchat_summary,
        wildchat_forms,
        offsets,
    )
    print(json.dumps({
        "output": str(args.out),
        "stanford_rows": len(stanford_rows),
        "wildchat": wildchat_summary,
        "formats": len(library),
        "formats_observed_in_wildchat": sum(item["wildchat_count"] > 0 for item in library),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
