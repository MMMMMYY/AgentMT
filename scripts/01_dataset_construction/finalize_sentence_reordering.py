"""Finalize conservative sentence-permutation variants for the 116-task suite."""
import sys as _sys, pathlib as _pl  # scripts are grouped by stage; make sibling stages importable
_sys.path[:0] = [str(_d) for _d in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _d.is_dir()]
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from prepare_sentence_reordering import split_sentences


# These exclusions are properties of sentence permutation, not model judgments.
# In each case, moving a complete sentence creates a forward reference or moves
# a prerequisite operation behind a sentence that depends on it.
EXCLUSIONS = {
    'AIP-026': 'label-it sentence would precede the payment it refers to',
    'AIP-035': 'He would precede its antecedent my son',
    'AIP-050': 'these would precede the movie list it refers to',
    'ACB-seed_calendar_titles_client_001': 'her would precede the named recipient Jane',
    'ACB-seed_env_secrets_001': 'him would precede the named recipient David',
    'ACB-seed_mom_dinner_calendar_001': 'Make sure the event would precede event creation',
    'TJB-Music-S-8-54': 'the first sentence is an operation required by later sentences',
    'TJB-News_Media-S-3-175': 'After getting the latest articles depends on the first sentence',
    'TJB-eCommerce-S-10-180': 'later this-product references depend on the first sentence',
}


def controlled_permutation(row: dict) -> tuple[list[str], list[str], str]:
    sentences = split_sentences(row['prompt'])
    if len(sentences) < 2:
        return sentences, sentences, 'not_applicable'
    if row['taskId'] in EXCLUSIONS:
        return sentences, sentences, 'excluded_dependency'
    if row['dataset'] == 'TRAJECT-Bench' and row.get('trajectoryType') == 'parallel' and len(sentences) > 2:
        # Preserve the preamble, rotate only independently executable requests.
        return sentences, [sentences[0], *sentences[2:], sentences[1]], 'parallel_action_rotation'
    # Move framing/context information after the request while preserving the
    # relative order of every remaining operation sentence.
    return sentences, [*sentences[1:], sentences[0]], 'context_to_end'


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    suites = args.output / 'suites'
    suites.mkdir(exist_ok=True)

    rows = json.loads(Path('dataset/environments/fixed_sources.json').read_text())['unified']
    records = []
    for row in rows:
        original, permuted, strategy = controlled_permutation(row)
        eligible = strategy not in {'not_applicable', 'excluded_dependency'}
        record = {
            'task_id': row['taskId'],
            'dataset': row['dataset'],
            'source_prompt': row['prompt'],
            'status': 'ready' if eligible else strategy,
            'strategy': strategy,
            'exclusion_reason': EXCLUSIONS.get(row['taskId']),
            'source_sentences': original,
            'variant': None,
            'independent_human_validation': False,
        }
        if eligible:
            record['variant'] = {
                'id': 'io01',
                'prompt': ' '.join(permuted),
                'transformation': 'sentence_permutation',
                'sentence_multiset_preserved': sorted(original) == sorted(permuted),
                'information_order_changed': original != permuted,
                'eligible_pilot': True,
                'validation_basis': 'deterministic span preservation and dependency exclusion rules',
            }
            task_suite = {
                'task_id': row['taskId'],
                'scenario_id': row['taskId'][4:] if row['taskId'].startswith('ACB-') else None,
                'status': 'candidates_generated',
                'variants': [record['variant']],
            }
            directory = suites / row['taskId']
            directory.mkdir(exist_ok=True)
            write_json(directory / 'suite_ready.json', task_suite)
            # Compatibility with the existing fixed-agent runners.
            write_json(directory / 'suite_ready_v3.json', task_suite)
        records.append(record)

    write_json(args.output / 'manifest.json', records)
    ready = [record for record in records if record['status'] == 'ready']
    (args.output / 'ready_pairs.jsonl').write_text(''.join(
        json.dumps(record, ensure_ascii=False) + '\n' for record in ready
    ))
    by_dataset = {}
    for record in ready:
        by_dataset[record['dataset']] = by_dataset.get(record['dataset'], 0) + 1
    summary = {
        'source_tasks': len(records),
        'multi_sentence_candidates_considered': sum(len(record['source_sentences']) > 1 for record in records),
        'ready_pairs': len(ready),
        'excluded_dependency': sum(record['status'] == 'excluded_dependency' for record in records),
        'not_applicable_single_sentence': sum(record['status'] == 'not_applicable' for record in records),
        'ready_by_dataset': by_dataset,
        'all_sentence_multisets_preserved': all(
            record['variant']['sentence_multiset_preserved'] for record in ready
        ),
        'independent_human_validation': False,
        'agent_experiments_run': False,
    }
    write_json(args.output / 'summary.json', summary)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
