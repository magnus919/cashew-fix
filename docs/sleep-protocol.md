# Sleep protocol integration

`core.sleep.run_sleep_cycle` is the bounded consolidation entry point. Existing
callers may continue to use the first three positional arguments; new adapter
options are keyword-only:

```python
run_sleep_cycle(
    db_path, limit=None, model_fn=None, *,
    background_dream=False, max_edges=100_000,
    cross_source_only=False, embedding_client=None,
    embedding_model=None, expected_dimension=None,
    journal_policy="manage",
)
```

Orphan repair is caller-owned. When repair is enabled, provide all three of
`embedding_client`, `embedding_model`, and `expected_dimension`. The client
must implement `encode(list[str])` and return a NumPy array with exactly one
finite, non-zero row per input and the requested dimension. Sleep does not
construct a model, select a device, or call a default embedding service.
Invalid output is rejected before any node write. Each valid node is written
to `embeddings` and `vec_embeddings` inside one savepoint. If the vec index is
present and rejects a write, the ordinary row is rolled back too. If the vec
index is genuinely absent, ordinary-only repair is reported by
`orphan_vec_unavailable`.

`journal_policy="manage"` retains the historical direct-call behavior.
Integrations that own SQLite admission should pass `journal_policy="preserve"`;
that mode performs no journal-mode assignment. The decay-audit table is created
idempotently on the cycle connection before any decay audit write.

The result is JSON-safe and includes `status`, bounded phase counters, dream
state (`skipped`, `pending`, `ran`, or `failed`), and explicit pair versus
directed-row counts. `cross_links_created` counts new unordered pairs only when
both directions are committed. A half-pair is repaired and counted separately;
existing complete pairs are skipped without consuming `max_edges`.
