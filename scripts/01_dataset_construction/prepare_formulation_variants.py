"""Prepare one auditable formulation-transformation family for 116 agent tasks.

Multi-sentence prompts use conservative sentence permutation, following the
SentenceReordering transformation in NL-Augmenter (and the sentence permutation
objective in BART).  Atomic prompts use the released ACL 2022 QCPG models.

This script generates candidates and applies deterministic surface guards.  It
does not claim that a neural paraphrase is semantically equivalent; accepted
QCPG candidates remain pending semantic/authorization review.
"""

from __future__ import annotations
import sys as _sys, pathlib as _pl  # scripts are grouped by stage; make sibling stages importable
_sys.path[:0] = [str(_d) for _d in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _d.is_dir()]

import argparse
from difflib import SequenceMatcher
import json
from pathlib import Path
import random
import re


SOURCE_FILE = Path("dataset/environments/fixed_sources.json")
REORDER_FILE = Path("dataset/variants/formulation_reordering/manifest.json")
CACHE_DIR = Path("vendor/huggingface-cache")
QUESTION_MODEL = "ibm-research/qcpg-questions"
SENTENCE_MODEL = "ibm-research/qcpg-sentences"
PRESERVED_NER_TYPES = {
    "PERSON", "ORGANIZATION", "LOCATION", "CITY", "STATE_OR_PROVINCE",
    "COUNTRY", "NATIONALITY", "MISC", "DATE", "TIME", "DURATION",
    "MONEY", "PERCENT", "ORDINAL", "NUMBER", "URL", "EMAIL",
}

NEGATION_SCOPE_TERMS = {
    "no", "not", "never", "only", "without", "except", "exact", "exactly",
}
MODAL_TERMS = {"can", "could", "would", "will", "should", "must", "need"}


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.casefold())


def protected_literals(text: str) -> list[str]:
    patterns = [
        r"https?://[^\s,;]+",
        r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b",
        r"#[\w-]+",
        # Double-quoted strings and genuinely quoted single-quoted strings.
        # The word-boundary guards prevent contractions such as `I'm ... Let's`
        # from being interpreted as one enormous quoted span.
        r'"[^"\n]+"|(?<!\w)\'[^\'\n]+\'(?!\w)',
        r"-?\d+(?:\.\d+)?,-?\d+(?:\.\d+)?",
        r"\$\s?\d[\d,]*(?:\.\d+)?",
        r"\b\d+(?:[.:,-]\d+)*(?:\s?(?:AM|PM|am|pm|%|m|s|min|h|day|week|month|year)s?)?\b",
        r"\b[A-Z][A-Z0-9_-]{1,}\b",
        r"\b(?:Open|Cloud|Online)[A-Z][A-Za-z0-9_-]*\b",
    ]
    found: list[str] = []
    for pattern in patterns:
        found.extend(re.findall(pattern, text))
    return sorted(set(found), key=lambda value: (text.find(value), value))


def term_set(text: str, vocabulary: set[str]) -> list[str]:
    return sorted(set(words(text)) & vocabulary)


def normalize(text: str) -> str:
    return " ".join(words(text))


def shield_literals(text: str) -> tuple[str, dict[str, str]]:
    """Replace permission-relevant literals before neural paraphrasing.

    Longer literals are replaced first so overlapping values such as
    ``$1,200`` and ``1,200`` cannot create nested placeholders.  Restoration
    is lossless; a missing placeholder remains a deterministic failure.
    """
    shielded = text
    restoration: dict[str, str] = {}
    for index, literal in enumerate(sorted(protected_literals(text), key=len, reverse=True)):
        if literal not in shielded:
            continue
        placeholder = f"PROTECTEDITEM{index}TOKEN"
        shielded = shielded.replace(literal, placeholder)
        restoration[placeholder] = literal
    return shielded, restoration


def restore_literals(text: str, restoration: dict[str, str]) -> str:
    restored = text
    for placeholder, literal in restoration.items():
        restored = re.sub(re.escape(placeholder), lambda _: literal, restored, flags=re.IGNORECASE)
    return restored


def controls(tokenizer, semantic: float, lexical: float, syntactic: float) -> list[str]:
    values = {
        "SEMANTIC_SIM": min(95, int(5 * round(semantic * 100 / 5))),
        "LEXICAL_DIV": min(100, int(5 * round(lexical * 100 / 5))),
        "SYNTACTIC_DIV": min(80, int(5 * round(syntactic * 100 / 5))),
    }
    tokens = [f"COND_{name}_{value}" for name, value in values.items()]
    missing = [token for token in tokens if token not in tokenizer.additional_special_tokens]
    if missing:
        raise ValueError(f"Tokenizer lacks QCPG controls: {missing}")
    return tokens


def deterministic_checks(source: str, candidate: str) -> dict:
    required = protected_literals(source)
    missing = [literal for literal in required if literal.casefold() not in candidate.casefold()]
    source_negation = term_set(source, NEGATION_SCOPE_TERMS)
    candidate_negation = term_set(candidate, NEGATION_SCOPE_TERMS)
    source_modals = term_set(source, MODAL_TERMS)
    candidate_modals = term_set(candidate, MODAL_TERMS)
    length_ratio = len(words(candidate)) / max(1, len(words(source)))
    similarity = SequenceMatcher(None, normalize(source), normalize(candidate)).ratio()
    failures = []
    if missing:
        failures.append("missing_protected_literal")
    if normalize(source) == normalize(candidate):
        failures.append("identity")
    if source.rstrip().endswith("?") != candidate.rstrip().endswith("?"):
        failures.append("question_force_changed")
    if source_negation != candidate_negation:
        failures.append("negation_or_scope_marker_changed")
    if not 0.70 <= length_ratio <= 1.35:
        failures.append("length_ratio_outside_guard")
    if similarity < 0.45:
        failures.append("surface_similarity_too_low")
    return {
        "passed": not failures,
        "failures": failures,
        "protected_literals": required,
        "missing_literals": missing,
        "source_negation_scope_terms": source_negation,
        "candidate_negation_scope_terms": candidate_negation,
        "source_modals": source_modals,
        "candidate_modals": candidate_modals,
        "length_ratio": round(length_ratio, 4),
        "surface_similarity": round(similarity, 4),
    }


def generate_group(rows, model_name, args, entities_by_task):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=args.cache_dir, local_files_only=True
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, cache_dir=args.cache_dir, local_files_only=True
    )
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device).eval()
    control_tokens = controls(tokenizer, args.semantic, args.lexical, args.syntactic)
    records = []
    for row in rows:
        generation_prompt = row["prompt"]
        restoration = {}
        if args.protect_literals:
            generation_prompt, restoration = shield_literals(generation_prompt)
        conditioned = " ".join(control_tokens) + generation_prompt
        encoded = tokenizer(conditioned, return_tensors="pt", truncation=True, max_length=512).to(device)
        preserved_terms = list(protected_literals(row["prompt"]))
        for entity in entities_by_task.get(row["taskId"], []):
            if entity.get("ner") in PRESERVED_NER_TYPES and entity["text"] not in preserved_terms:
                preserved_terms.append(entity["text"])
        # Remove shorter overlapping constraints. Requiring both `$1,200` and
        # `1,200`, for example, can force a duplicate phrase into the output.
        preserved_terms = [
            term for term in sorted(preserved_terms, key=len, reverse=True)
            if not any(term in longer for longer in preserved_terms if len(longer) > len(term))
        ]
        force_words_ids = [
            tokenizer.encode(term, add_special_tokens=False)
            for term in preserved_terms
            if tokenizer.encode(term, add_special_tokens=False)
        ]
        with torch.inference_mode():
            if args.num_candidates == 1:
                # Match the released QCPG usage: greedy single-output
                # generation.  The authors' prediction script defaults the
                # maximum target length to 128.
                output = model.generate(**encoded, max_length=128)
            else:
                generation = {
                    "num_return_sequences": args.num_candidates,
                    "max_length": 128,
                }
                if args.decoding == "sample":
                    generation.update(do_sample=True, top_p=.95, temperature=1.0)
                else:
                    generation.update(
                        do_sample=False,
                        num_beams=max(args.num_candidates, 12),
                    )
                if args.force_preserved_terms and force_words_ids:
                    generation.update(
                        do_sample=False,
                        num_beams=max(args.num_candidates, 12),
                        force_words_ids=force_words_ids,
                    )
                output = model.generate(**encoded, **generation)
        generated = tokenizer.batch_decode(output, skip_special_tokens=True)
        if restoration:
            generated = [restore_literals(value, restoration) for value in generated]
        unique = list(dict.fromkeys(value.strip() for value in generated if value.strip()))
        candidates = []
        for index, text in enumerate(unique):
            check = deterministic_checks(row["prompt"], text)
            candidates.append({
                "candidate_id": f"q{index + 1:02d}",
                "prompt": text,
                "deterministic_checks": check,
                "semantic_equivalence_reviewed": False,
                "authorization_equivalence_reviewed": False,
            })
        candidates.sort(
            key=lambda item: (
                not item["deterministic_checks"]["passed"],
                -item["deterministic_checks"]["surface_similarity"],
                item["candidate_id"],
            )
        )
        records.append({
            "task_id": row["taskId"],
            "dataset": row["dataset"],
            "source_prompt": row["prompt"],
            "transformation": "syntactic_paraphrasing",
            "method": "QCPG",
            "model": model_name,
            "controls": control_tokens,
            "literal_protection": {
                "enabled": args.protect_literals,
                "restoration_map": restoration,
            },
            "constrained_decoding": {
                "enabled": args.force_preserved_terms,
                "preserved_terms": preserved_terms if args.force_preserved_terms else [],
            },
            "candidates": candidates,
        })
        passed = sum(candidate["deterministic_checks"]["passed"] for candidate in candidates)
        print(f'{row["taskId"]}: {len(candidates)} candidates, {passed} pass surface guards', flush=True)
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return records


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--num-candidates", type=int, default=1)
    parser.add_argument("--decoding", choices=("beam", "sample"), default="beam")
    parser.add_argument("--protect-literals", action="store_true")
    parser.add_argument("--force-preserved-terms", action="store_true")
    parser.add_argument(
        "--entity-file",
        type=Path,
        default=Path("dataset/variants/locked_spans_ner/entities.json"),
    )
    # QCPG requires explicit control values and publishes no default triple.
    # These are the values in the authors' official README usage example.
    parser.add_argument("--semantic", type=float, default=.80)
    parser.add_argument("--lexical", type=float, default=.30)
    parser.add_argument("--syntactic", type=float, default=.50)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument(
        "--only-task-ids",
        type=Path,
        help="Optional JSON list of task IDs; useful for conservative retry batches.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    if args.num_candidates > 1:
        import torch
        torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = json.loads(SOURCE_FILE.read_text())["unified"]
    if args.only_task_ids:
        requested = set(json.loads(args.only_task_ids.read_text()))
        rows = [row for row in rows if row["taskId"] in requested]
    row_by_id = {row["taskId"]: row for row in rows}
    entities_by_task = {}
    if args.entity_file.exists():
        entities_by_task = {
            record["task_id"]: record["entities"]
            for record in json.loads(args.entity_file.read_text())
        }
    reorder_manifest = json.loads(REORDER_FILE.read_text())
    reorder_records = []
    paraphrase_ids = []
    for record in reorder_manifest:
        if record["task_id"] not in row_by_id:
            continue
        if record["status"] == "ready":
            reorder_records.append({
                "task_id": record["task_id"],
                "dataset": record["dataset"],
                "source_prompt": record["source_prompt"],
                "transformation": "information_reordering",
                "method": "NL-Augmenter-style sentence permutation with dependency exclusions",
                "candidate": record["variant"],
                "semantic_equivalence_basis": "Every complete sentence is preserved verbatim; dependency exclusions prevent known forward-reference and prerequisite inversions.",
                "authorization_equivalence_reviewed": False,
            })
        else:
            paraphrase_ids.append(record["task_id"])

    paraphrase_rows = [row_by_id[task_id] for task_id in paraphrase_ids]
    question_rows = [row for row in paraphrase_rows if row["prompt"].rstrip().endswith("?")]
    sentence_rows = [row for row in paraphrase_rows if not row["prompt"].rstrip().endswith("?")]
    paraphrase_records = []
    paraphrase_records.extend(generate_group(question_rows, QUESTION_MODEL, args, entities_by_task))
    paraphrase_records.extend(generate_group(sentence_rows, SENTENCE_MODEL, args, entities_by_task))

    all_records = sorted(reorder_records + paraphrase_records, key=lambda item: item["task_id"])
    write_json(args.output / "candidates.json", all_records)
    auto_pass = sum(
        any(candidate["deterministic_checks"]["passed"] for candidate in record["candidates"])
        for record in paraphrase_records
    )
    summary = {
        "source_tasks": len(rows),
        "information_reordering_tasks": len(reorder_records),
        "syntactic_paraphrasing_tasks": len(paraphrase_records),
        "qcpg_question_tasks": len(question_rows),
        "qcpg_sentence_tasks": len(sentence_rows),
        "qcpg_tasks_with_surface_guard_candidate": auto_pass,
        "qcpg_tasks_without_surface_guard_candidate": len(paraphrase_records) - auto_pass,
        "total_formulation_candidates": len(all_records),
        "reordering_candidate_count": len(reorder_records),
        "qcpg_candidate_count": sum(len(record["candidates"]) for record in paraphrase_records),
        "semantic_review_complete": False,
        "authorization_review_complete": False,
        "formal_variants_finalized": False,
        "seed": args.seed,
        "parameter_policy": {
            "generation": (
                "greedy single output; official prediction max target length 128"
                if args.num_candidates == 1
                else f"{args.decoding} decoding; {args.num_candidates} outputs; maximum target length 128"
            ),
            "quality_controls": "official QCPG README usage example (no default triple is published)",
            "semantic": args.semantic,
            "lexical": args.lexical,
            "syntactic": args.syntactic,
            "permission_relevant_literal_protection": args.protect_literals,
            "forced_preserved_terms": args.force_preserved_terms,
            "named_entity_source": str(args.entity_file) if args.force_preserved_terms else None,
        },
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
