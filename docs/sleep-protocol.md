# Sleep protocol integration

`core.sleep.run_sleep_cycle` is the bounded consolidation entry point. Existing
callers may continue to use the first six positional arguments; new adapter
options are keyword-only:

```python
run_sleep_cycle(
    db_path, limit=None, model_fn=None, background_dream=False,
    max_edges=100_000, cross_source_only=False, *, embedding_client=None,
    embedding_model=None, expected_dimension=None,
    journal_policy="manage", orphan_limit=None, orphan_batch_size=100,
)
```

Orphan repair is caller-owned. When repair is enabled, provide all three of
`embedding_client`, `embedding_model`, and `expected_dimension`. The client
must implement `encode(list[str])` and return a NumPy array with exactly one
finite, non-zero row per input and the requested dimension. Sleep does not
construct a model, select a device, or call a default embedding service.
Invalid output is rejected before any node write. Each valid node is written
to `embeddings` and `vec_embeddings` inside one savepoint, and each batch is
committed independently. If the vec index is present and rejects a write, the
ordinary row is rolled back too. If the vec index is genuinely absent,
ordinary-only repair is reported by `orphan_vec_unavailable`. The default
repairs all eligible rows in batches of 100 for backward compatibility.
`orphan_limit` caps rows examined across ordinary and vec-index repair;
`orphan_batch_size` may reduce the encode/commit batch below 100. A table merely
named `vec_embeddings` is not a vec capability: it must be a `vec0` virtual
table that can be loaded and queried on the active connection.

When `limit` caps candidate discovery, sleep persists a private cursor and
rotates deterministically through `(timestamp, node_id)` pages. Cursor claims
commit before expensive phases, so a stopped cycle may defer its page until
the next wrap but cannot keep later pages permanently starved. Node timestamps
remain untouched. The private cursor table is checked on every capped run;
partial or malformed pre-release shapes are rebuilt transactionally, retaining
a single type-valid cursor when possible. Duplicate or invalid legacy cursor
rows reset to the deterministic origin. Capped orphan work has separate durable
ordinary and vec-repair cursors and alternates which phase receives the first
share of the cap. A repeatedly failing oldest batch therefore cannot starve
later rows or the other repair class forever.

## Supported work envelope

The public limits bound specific phases; they are not an end-to-end deadline:

- `limit=N` bounds the candidate embedding matrix to at most `N` active nodes.
  Similarity memory and arithmetic are quadratic in `N`.
- `max_edges=M` bounds newly completed or repaired unordered cross-link pairs.
  Existing, same-source, or failed candidates do not consume the budget, so
  candidate inspection can still visit every pair in the `N`-node matrix.
- deduplication consumes the same bounded candidate page, but maximal-clique
  enumeration and edge rewiring do not currently have a separate time or write
  budget.
- GC mutates at most 50 sampled nodes and orphan embedding examines at most
  `orphan_limit` rows when supplied. Metrics, permanence, core-memory ranking,
  audit cleanup, and vec compaction can still scan graph-wide state.
- a synchronous `model_fn` has no engine-enforced deadline. Integrations that
  need a wall-clock bound must own cancellation and SQLite admission outside
  this function.

Dream generation only receives cross-link pairs completed or repaired and
committed by the current cycle. Capped, filtered, failed, and already-complete
pairs are excluded. In particular, `max_edges=0` cannot invoke the dream model.

For reproducible local measurements, `scripts/benchmark_sleep_cycle.py` builds
an isolated synthetic database and reports wall time plus the public counters.
It never opens the configured Cashew database. `--hold-writer-ms` adds a
deterministic competing write transaction before the cycle starts. Results
characterize the chosen machine and fixture; they do not establish a
production deadline.

`journal_policy="manage"` retains the historical direct-call behavior.
Integrations that own SQLite admission should pass `journal_policy="preserve"`;
that mode performs no journal-mode assignment. The decay-audit table is created
idempotently on the cycle connection before any decay audit write.

The result is JSON-safe and includes `status`, bounded phase counters, dream
state (`skipped`, `pending`, `ran`, or `failed`), and explicit pair versus
directed-row counts. `cross_links_created` counts new unordered pairs only when
both directions are committed. A half-pair is repaired and counted separately;
existing complete pairs are skipped without consuming `max_edges`.
Cross-link counters and dream inputs advance only after each batch commit. A
known rolled-back suffix reports the exact committed prefix as `partial`; if a
commit's durability cannot be verified, the stable result instead reports
`uncertain` without claiming the attempted batch.
