# Three-benchmark environment integration — partial, not experiment-ready

Historical preparation report. The current implementation and actual pilot
results are described in `../FIXED_EXPERIMENT_STATUS_20260909.md`. The AgentCI
text-tool adapter has since been implemented; that is distinct from launching
the original visual OpenApps runtime. All 116 environments are still not ready.

## Live service update

`privacy_mt/traject_live.py` implements the official ToolBench-compatible POST
payload using caller-configured `API_URL` and `TOOLBENCH_KEY`. HTTPS is required;
redirects and automatic retries are disabled. Execution requires an explicitly
reviewed tool allowlist. Never silently fall back to recorded responses.

No live connection has been verified: both configuration variables were absent.
Do not paste credentials into chat or commit them. A Gemini key is not a
ToolBench service credential. Supply service credentials through the local
environment, then run:

```sh
.conda-gemini-agent/bin/python scripts/smoke_traject_live.py
```

The smoke test only queries a mathematical fact about 17. A successful response
verifies this one endpoint/tool, not availability of all 263 reference calls.
The full catalog contains duplicate tool names; construct and validate scoped
tool pools before full experiments, rather than resolving ambiguous names by
silently picking the first record. Full task execution remains unverified.

This directory freezes the selected 40 AIAP + 36 AgentCI + 40 TRAJECT source
records. It contains no new model experiment results.

## Implemented

- AIAP permission gate: source candidate schema only; fixture reads require a
  value and provenance. Missing data produces an environment error, never a
  grant or invented result. Dataset relevance labels are not exposed by catalog.
- AgentCI: installed hydra-core 1.3.2; prepared official OpenApps runtime configs
  for each selected seed. A local compatibility subclass preserves the empty
  destination todo list in seed_chat_household_chores_001. Original sources stay
  unchanged. Conversion warnings are preserved in readiness.json.
- TRAJECT: exact argument-matched replay of published responses. This is a
  smoke-test backend only, NOT a live executor or a valid unrestricted path-
  expansion experiment. Unrecorded or conflicting calls fail explicitly.
- Event logs distinguish permission requests, sandbox grants, actual fixture
  reads, recorded tool results, and environment errors.

## Still required before the 116-task model run

1. AIAP: map published fixed values for each selected task and implement its
   business actions and success evaluator. Category labels alone are not files
   or account records. No arbitrary fixture replacement was made.
2. AgentCI: launch and test the OpenApps UI runtime; connect the agent action
   loop and evaluate observable state changes. Generated configs do not prove
   successful server startup or information-preserving UI conversion.
3. TRAJECT: configure an authorized tool service, or explicitly choose and
   document a restricted offline experimental design. Do not use credentials
   embedded in third-party source code. API_URL and TOOLBENCH_KEY were absent
   from the checked process environment; no remote service call was made.
4. Connect model runner, step budgets, episode reset, outcome evaluators, and
   complete event instrumentation. No gold tool sequence may be used as the
   agent's entire available tool catalog.

Run from repository root:

```sh
.conda-gemini-agent/bin/python scripts/prepare_benchmark_environments.py
.conda-gemini-agent/bin/python -m unittest tests.test_benchmark_environments
```

`fixed_sources.json` is the immutable input snapshot for this preparation;
`readiness.json` includes its content digest and per-task state. Raw smoke-test
events are in `replay_smoke/`. Runtime configs and adapter metadata are in
`agentci_runtimes/`. No claim of minimal-path ground truth or actual privacy
violation follows from passing these smoke tests.
