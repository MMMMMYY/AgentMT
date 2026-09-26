# Label-free prompt-style surface extraction

## Objective

The extraction stage discovers recurrent ways in which users formulate
requests without assigning style labels and without searching for a predefined
list of polite, neutral, or hostile constructions. The output is therefore a
frequency inventory of **surface-form candidates**, not a finalized style
taxonomy.

## Input unit

Every user turn in every conversation is processed, including initial prompts,
follow-up requests, corrections, and retries. Assistant turns are excluded.
The dataset-provided language field is retained for each user turn.

## Processing pipeline

### 1. Streaming

The Hugging Face train split is read as a stream. Conversations are processed
in groups of 10,000 and are never collected into one in-memory corpus. Each
completed group produces an aggregate checkpoint, so an interrupted run can
resume.

### 2. Privacy-preserving normalization

Before counting, fenced code blocks are removed from the linguistic inventory;
URLs, email addresses, and long numbers are replaced with placeholders; and
whitespace is normalized. Complete user messages are not written to disk.

### 3. Exact-turn deduplication

The normalized user turn is case-folded and represented by a SHA-256 digest.
If the same complete normalized turn occurs again anywhere in the corpus, it is
skipped. Only the digest is stored in the resumable working data. This reduces
the effect of copied prompt templates and repeated dataset records.

### 4. Multilingual boundary tokenization

Unicode letter sequences are retained as tokens. Chinese ideographs are treated
as individual tokens; Japanese kana and Korean Hangul sequences are retained.
This is a language-agnostic surface tokenizer rather than a language-specific
morphological analyzer.

### 5. Boundary extraction

For each unique user turn, four observable positions are extracted:

- `prompt_start`: prefixes of the complete user turn;
- `prompt_end`: suffixes of the complete user turn;
- `sentence_start`: prefixes of each sentence or newline-delimited segment;
- `sentence_end`: suffixes of each sentence or segment.

Prefixes of one through eight tokens and suffixes of one through six tokens are
recorded. At most the first 12 sentence segments of a user turn are processed.
Within one user turn, the same `(language, position, length, phrase)` tuple
contributes at most one count. Consequently, `turn_count` measures the number of
distinct deduplicated user turns containing the boundary form, rather than the
raw number of repetitions inside a long prompt.

Formally, for a boundary form \(f\), its corpus frequency is

\[
C(f) = \sum_{u \in U^*} \mathbf{1}[f \in B(u)],
\]

where \(U^*\) is the set of exact-deduplicated user turns and \(B(u)\) is the
set of boundary forms extracted from turn \(u\).

### 6. Exact global aggregation

Each 10,000-conversation checkpoint stores sorted aggregate counts, not raw
messages. After all rows are processed, a multiway merge sums identical tuples
across checkpoints. This produces exact corpus-level frequencies rather than an
approximate heavy-hitter estimate.

### 7. Frequency and release threshold

Only forms occurring in at least five distinct deduplicated user turns are
released. This removes singletons and some accidental personal fragments while
retaining the long tail. Retained forms are sorted by `turn_count` from highest
to lowest. No politeness, tone, request-intent, or toxicity class is assigned in
this stage.

## Output schema

The complete table contains:

- `frequency_rank`;
- `language`;
- `position`;
- `n_tokens`;
- `phrase`;
- `turn_count`.

## What this method can and cannot establish

This method gives a reproducible census of recurring boundary realizations in
the selected corpus. It does not yet determine which phrases express the same
style, whether a phrase is a request, or whether two complete prompts preserve
task and authorization semantics. Those questions belong to the later
abstraction, classification, and metamorphic-prompt validation stages.

Important limitations are that boundary n-grams can contain topical rather
than stylistic content; exact deduplication does not remove paraphrased or
near-duplicate boilerplate; language labels come from the dataset; and the
language-agnostic tokenizer is not equivalent to proper segmentation for every
language.

## Applying the method to WildChat-1M-Full

`allenai/WildChat-1M-Full` is gated because it contains potentially harmful and
sensitive content. Once the Hugging Face account has been granted access and
the local environment is authenticated, the same extractor can be run with:

```bash
python3 -m privacy_mt.full_wildchat_style_extraction \
  --dataset allenai/WildChat-1M-Full \
  --out data/prompt_style_formats/full_wildchat_1m_sensitive_v1 \
  --chunk-rows 10000 \
  --min-count 5 \
  --top-n 50000
```

