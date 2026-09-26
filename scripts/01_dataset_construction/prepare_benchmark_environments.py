"""Prepare official AgentCI runtimes and test all selected recorded tool calls."""
import sys as _sys, pathlib as _pl  # scripts are grouped by stage; make sibling stages importable
_sys.path[:0] = [str(_d) for _d in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _d.is_dir()]
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from privacy_mt.benchmark_environments import TrajectoryReplayEnvironment, parameters, fixture_digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agentci-root', type=Path, default=Path('/private/tmp/agentcibench_repo'))
    parser.add_argument('--output', type=Path, default=ROOT / 'dataset/environments')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'dataset/source_selection/selected_sources.json'
    selected = json.loads(source.read_text())
    # Freeze all selected records before any model execution.
    (args.output / 'fixed_sources.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2))
    sys.path.insert(0, str(args.agentci_root))
    from envs.openapps_wrapper import OpenAppsWrapper
    class EmptyTodoCompatibleWrapper(OpenAppsWrapper):
        # Official seed contains an intentionally empty destination list.
        # Preserve it; upstream's truthiness check incorrectly rejects it.
        def _convert_todo(self, state):
            if isinstance(state, dict) and state.get('items') == [] and not state.get('todo'):
                return []
            return super()._convert_todo(state)

    wrapper = EmptyTodoCompatibleWrapper(args.output / 'agentci_runtimes')
    report = {'source_sha256': fixture_digest(selected), 'model_calls': 0,
              'full_experiment_ready': False, 'agentci': [], 'traject': [], 'aiap': []}
    for scenario in selected['agentCiRows']:
        row = {'id': scenario['scenario_id']}
        try:
            runtime = wrapper.prepare_runtime(scenario, run_id=scenario['scenario_id'], strict=True)
            row.update(status='official_config_prepared_not_launched', config=str(runtime.config_path),
                       warnings=runtime.warnings,
                       local_adapter='Preserves empty destination todo list; otherwise upstream converter')
        except Exception as exc:
            row.update(status='blocked', error=str(exc))
        report['agentci'].append(row)
    for index, trajectory in enumerate(selected['trajectRows']):
        env = TrajectoryReplayEnvironment(trajectory)
        row = {'index': index, 'status': 'recorded_replay_only', 'calls': 0}
        try:
            for tool in trajectory['tool list']:
                env.call(tool['tool name'], parameters(tool))
                row['calls'] += 1
        except Exception as exc:
            row.update(status='blocked', error=str(exc))
        env.save(args.output / 'replay_smoke' / f'{index:02d}.jsonl')
        report['traject'].append(row)
    for task in selected['aiapRows']:
        report['aiap'].append({'id': task['id'], 'status': 'blocked',
            'reason': 'Published value fixtures and task-specific completion executor not mapped'})
    (args.output / 'readiness.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: {status: sum(r['status'] == status for r in report[k])
                        for status in sorted({r['status'] for r in report[k]})}
                      for k in ['agentci','traject','aiap']}))


if __name__ == '__main__':
    main()
