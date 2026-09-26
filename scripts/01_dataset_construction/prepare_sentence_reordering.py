"""Create deterministic sentence-permutation MT variants.

The transformation follows the sentence permutation noise used by BART and the
SentenceReordering transformation in NL-Augmenter.  It deliberately does not
rewrite sentence contents: only complete sentence spans are permuted.
"""
import sys as _sys, pathlib as _pl  # scripts are grouped by stage; make sibling stages importable
_sys.path[:0] = [str(_d) for _d in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _d.is_dir()]
import argparse
import hashlib
import json
from pathlib import Path
import random
import re

try:
    import nltk
except ImportError:  # Keep the script usable in the minimal harness environment.
    nltk = None


def split_sentences(text: str) -> list[str]:
    if nltk is not None:
        return nltk.sent_tokenize(text)
    # Dependency-light fallback. Protect common title abbreviations occurring in
    # the fixed benchmark before splitting at sentence-final punctuation.
    protected = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|St)\.', r'\1<DOT>', text)
    return [
        part.replace('<DOT>', '.').strip()
        for part in re.split(r'(?<=[.!?])\s+(?=["“”\'A-Z])', protected)
        if part.strip()
    ]


def permute_sentences(text: str, seed: int) -> tuple[list[str], list[str]]:
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return sentences, sentences
    order = list(range(len(sentences)))
    random.Random(seed).shuffle(order)
    if order == list(range(len(sentences))):
        order = order[1:] + order[:1]
    return sentences, [sentences[index] for index in order]


def stable_seed(task_id: str, base_seed: int) -> int:
    digest = hashlib.sha256(task_id.encode('utf-8')).digest()
    return base_seed + int.from_bytes(digest[:8], 'big')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    source_path = Path('dataset/environments/fixed_sources.json')
    rows = json.loads(source_path.read_text())['unified']
    records = []
    for row in rows:
        original, permuted = permute_sentences(
            row['prompt'], stable_seed(row['taskId'], args.seed)
        )
        applicable = len(original) > 1
        candidate = ' '.join(permuted) if applicable else None
        records.append({
            'task_id': row['taskId'],
            'dataset': row['dataset'],
            'source_prompt': row['prompt'],
            'source_sentences': original,
            'status': 'candidate_generated' if applicable else 'not_applicable',
            'not_applicable_reason': None if applicable else 'fewer_than_two_sentences',
            'candidate': None if not applicable else {
                'variant_id': 'sentence_permutation_01',
                'text': candidate,
                'sentence_multiset_preserved': sorted(original) == sorted(permuted),
                'task_equivalence_reviewed': False,
                'authorization_equivalence_reviewed': False,
                'information_order_changed': original != permuted,
            },
        })

    (args.output / 'families.json').write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + '\n'
    )
    candidates = [record for record in records if record['candidate']]
    (args.output / 'candidates.jsonl').write_text(''.join(
        json.dumps(record, ensure_ascii=False) + '\n' for record in candidates
    ))
    summary = {
        'method': 'deterministic sentence permutation',
        'method_precedent': [
            'BART sentence permutation (Lewis et al., ACL 2020)',
            'NL-Augmenter SentenceReordering',
        ],
        'source_tasks': len(records),
        'applicable_tasks': len(candidates),
        'not_applicable_tasks': len(records) - len(candidates),
        'variants': len(candidates),
        'all_sentence_multisets_preserved': all(
            record['candidate']['sentence_multiset_preserved'] for record in candidates
        ),
        'seed': args.seed,
        'coreference_resolution': False,
        'human_validated': False,
        'agent_experiments_run': False,
    }
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
