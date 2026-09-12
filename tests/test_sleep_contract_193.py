"""Production-boundary contracts for the upstream sleep adapter."""

from __future__ import annotations

import sqlite3
import inspect
from pathlib import Path

import numpy as np
import pytest

from core.decay_audit import ensure_decay_audit_schema
import core.sleep as sleep_module
from core.sleep import (
    _batch_cross_links, _embed_orphans, _empty_sleep_result,
    _vec_write_capability, run_sleep_cycle,
)


class _CountingConnection(sqlite3.Connection):
    commit_count = 0

    def commit(self):
        self.commit_count += 1
        return super().commit()


def _edge_db(tmp_path: Path, name: str = "edges.db", factory=None) -> sqlite3.Connection:
    kwargs = {"factory": factory} if factory is not None else {}
    conn = sqlite3.connect(str(tmp_path / name), **kwargs)
    conn.executescript("""
        CREATE TABLE derivation_edges(parent_id TEXT, child_id TEXT,
            weight REAL, reasoning TEXT, PRIMARY KEY(parent_id, child_id));
    """)
    return conn


def _cycle_db(tmp_path: Path, name: str = "cycle.db") -> Path:
    path = tmp_path / name
    conn = sqlite3.connect(str(path))
    conn.executescript("""
        CREATE TABLE thought_nodes(
            id TEXT PRIMARY KEY, content TEXT, decayed INTEGER DEFAULT 0,
            timestamp TEXT DEFAULT '', source_file TEXT, access_count INTEGER DEFAULT 0,
            permanent INTEGER DEFAULT 0, last_accessed TEXT, domain TEXT, node_type TEXT,
            confidence REAL, metadata TEXT, mood_state TEXT
        );
        CREATE TABLE embeddings(
            node_id TEXT PRIMARY KEY, vector BLOB, model TEXT, updated_at TEXT
        );
        CREATE TABLE derivation_edges(
            parent_id TEXT, child_id TEXT, weight REAL, reasoning TEXT,
            PRIMARY KEY(parent_id, child_id)
        );
    """)
    vector = np.ones(1024, dtype=np.float32).tobytes()
    for nid in ("a", "b"):
        conn.execute("INSERT INTO thought_nodes(id, content) VALUES (?, ?)", (nid, nid))
        conn.execute(
            "INSERT INTO embeddings(node_id, vector, model) VALUES (?, ?, ?)",
            (nid, vector, "thenlper/gte-large"),
        )
    conn.commit()
    conn.close()
    return path


def test_cross_link_cap_flushes_two_directed_rows(tmp_path):
    conn = _edge_db(tmp_path)
    stats = _batch_cross_links(
        conn, ["a", "b"], np.array([[0, 1]]), np.eye(2), max_edges=1
    )
    assert stats["created"] == 1
    assert stats["directed_rows"] == 2
    assert conn.execute("SELECT count(*) FROM derivation_edges").fetchone()[0] == 2
    conn.close()


def test_cross_link_repairs_half_pair_and_counts_one_row(tmp_path):
    conn = _edge_db(tmp_path)
    conn.execute("INSERT INTO derivation_edges VALUES ('a','b',.9,'old')")
    conn.commit()
    stats = _batch_cross_links(
        conn, ["a", "b"], np.array([[0, 1]]), np.eye(2), max_edges=1
    )
    assert stats["created"] == 0 and stats["repaired"] == 1
    assert stats["directed_rows"] == 1
    assert conn.execute("SELECT count(*) FROM derivation_edges").fetchone()[0] == 2
    conn.close()


def test_cross_link_trigger_suppression_is_not_claimed(tmp_path):
    conn = _edge_db(tmp_path)
    conn.execute(
        """CREATE TRIGGER suppress_reverse BEFORE INSERT ON derivation_edges
           WHEN NEW.parent_id='b' AND NEW.child_id='a'
           BEGIN SELECT RAISE(IGNORE); END"""
    )
    conn.commit()
    stats = _batch_cross_links(
        conn, ["a", "b"], np.array([[0, 1]]), np.eye(2), max_edges=1
    )
    assert stats["created"] == 0
    assert stats["failed"] == 1
    assert conn.execute("SELECT count(*) FROM derivation_edges").fetchone()[0] == 0
    conn.close()


@pytest.mark.parametrize("pair_count", [500, 501])
def test_cross_link_cap_batches_at_five_hundred_pairs(tmp_path, pair_count):
    ids = [f"n{i}" for i in range(pair_count * 2)]
    pairs = np.asarray([[2 * i, 2 * i + 1] for i in range(pair_count)])
    sim = np.eye(len(ids), dtype=np.float32)
    conn = _edge_db(tmp_path, name=f"edges-{pair_count}.db", factory=_CountingConnection)
    stats = _batch_cross_links(conn, ids, pairs, sim, max_edges=pair_count)
    assert stats["created"] == pair_count
    assert stats["directed_rows"] == pair_count * 2
    assert conn.execute("SELECT count(*) FROM derivation_edges").fetchone()[0] == pair_count * 2
    assert conn.commit_count <= 2
    conn.close()


def test_run_sleep_cycle_preserves_six_positional_arguments():
    signature = inspect.signature(run_sleep_cycle)
    signature.bind("/tmp/example.db", None, None, False, 1, True)


def test_partial_embedding_triad_is_rejected_before_database_open(monkeypatch):
    def fail_connect(*_args, **_kwargs):
        raise AssertionError("database must not open during argument rejection")

    monkeypatch.setattr(sqlite3, "connect", fail_connect)
    result = run_sleep_cycle(db_path="/tmp/does-not-exist.db", embedding_model="m")
    assert result["status"] == "rejected"
    assert result["error"] == "invalid_embedding_contract"


def test_embedding_profile_dimension_is_rejected_before_database_open(monkeypatch):
    def fail_connect(*_args, **_kwargs):
        raise AssertionError("database must not open for profile mismatch")

    monkeypatch.setattr(sqlite3, "connect", fail_connect)
    result = run_sleep_cycle(
        db_path="/tmp/does-not-exist.db",
        embedding_client=_Client(lambda n: np.ones((n, 4), dtype=np.float32)),
        embedding_model="all-MiniLM-L6-v2",
        expected_dimension=4,
    )
    assert result["status"] == "rejected"
    assert result["error"] == "embedding_dimension_mismatch"


def test_active_profile_failure_without_explicit_model_is_pre_db(monkeypatch):
    monkeypatch.setattr(sleep_module, "_get_active_profile", lambda *_args: (_ for _ in ()).throw(RuntimeError("profile")))
    monkeypatch.setattr(sqlite3, "connect", lambda *_args, **_kwargs: pytest.fail("database must not open"))
    result = run_sleep_cycle(db_path="/tmp/does-not-exist.db")
    assert result["status"] in {"rejected", "unavailable"}
    assert result["error"] == "uncalibrated_embedding_model"


class _Client:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def encode(self, texts):
        self.calls.append(texts)
        return self.value(len(texts)) if callable(self.value) else self.value


def _orphan_db(tmp_path: Path, vec: bool = True):
    conn = sqlite3.connect(str(tmp_path / "orphans.db"))
    conn.execute(
        "CREATE TABLE thought_nodes("
        "id TEXT PRIMARY KEY, content TEXT, decayed INTEGER DEFAULT 0, "
        "timestamp TEXT DEFAULT '', source_file TEXT, access_count INTEGER DEFAULT 0, "
        "permanent INTEGER DEFAULT 0, node_type TEXT DEFAULT 'observation', "
        "last_accessed TEXT, domain TEXT)"
    )
    conn.execute(
        "CREATE TABLE embeddings("
        "node_id TEXT PRIMARY KEY, vector BLOB, model TEXT, updated_at TEXT)"
    )
    conn.execute(
        "CREATE TABLE derivation_edges("
        "parent_id TEXT, child_id TEXT, weight REAL, reasoning TEXT, "
        "PRIMARY KEY(parent_id, child_id))"
    )
    if vec:
        conn.execute(
            "CREATE TABLE vec_embeddings(node_id TEXT PRIMARY KEY, embedding BLOB)"
        )
    conn.execute(
        "INSERT INTO thought_nodes (id, content, decayed) VALUES ('n1','orphan',0)"
    )
    conn.commit()
    return conn


def test_orphan_invalid_batch_writes_nothing(tmp_path):
    conn = _orphan_db(tmp_path)
    client = _Client(lambda n: np.array([[np.nan, 1, 1, 1]], dtype=np.float32))
    stats = {}
    assert (
        _embed_orphans(
            conn,
            embedding_client=client,
            embedding_model="m",
            expected_dimension=4,
            stats=stats,
        )
        == 0
    )
    assert conn.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 0
    assert stats["orphan_write_failed"] == 1
    conn.close()


def test_orphan_vec_failure_rolls_back_ordinary_row(tmp_path):
    conn = _orphan_db(tmp_path)
    conn.execute("DROP TABLE vec_embeddings")
    conn.execute(
        "CREATE TABLE vec_embeddings("
        "node_id TEXT PRIMARY KEY, embedding BLOB NOT NULL CHECK(length(embedding)<1))"
    )
    conn.commit()
    client = _Client(lambda n: np.ones((n, 4), dtype=np.float32))
    stats = {}
    assert (
        _embed_orphans(
            conn,
            embedding_client=client,
            embedding_model="m",
            expected_dimension=4,
            stats=stats,
        )
        == 0
    )
    assert conn.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 0
    assert stats["orphan_write_failed"] == 1
    conn.close()


def test_orphan_repair_works_without_two_anchors(tmp_path):
    conn = _orphan_db(tmp_path)
    vec = np.ones(4, dtype=np.float32).tobytes()
    conn.execute(
        "INSERT INTO embeddings VALUES ('n1', ?, 'm', datetime('now'))", (vec,)
    )
    conn.commit()
    # Repair is deliberately skipped without the explicit client/model/dim
    # triad; sleep must not borrow a configured model by accident.
    stats = {}
    assert _embed_orphans(conn, expected_dimension=4, stats=stats) == 0
    assert stats["capability_missing"] is True
    assert conn.execute("SELECT count(*) FROM vec_embeddings").fetchone()[0] == 0
    conn.close()


def test_orphan_without_vec_is_explicit_ordinary_only(tmp_path):
    conn = _orphan_db(tmp_path, vec=False)
    client = _Client(lambda n: np.ones((n, 4), dtype=np.float32))
    stats = {}
    assert (
        _embed_orphans(
            conn,
            embedding_client=client,
            embedding_model="m",
            expected_dimension=4,
            stats=stats,
        )
        == 0
    )
    assert stats["orphan_vec_unavailable"] == 1
    conn.close()


def test_malformed_existing_embedding_is_contained(tmp_path):
    conn = _orphan_db(tmp_path)
    conn.execute(
        "INSERT INTO embeddings VALUES ('n1', ?, 'm', datetime('now'))", (b"bad",)
    )
    conn.commit()
    stats = {}
    assert _embed_orphans(
        conn, embedding_client=_Client(lambda n: np.ones((n, 4), dtype=np.float32)),
        embedding_model="m", expected_dimension=4, stats=stats,
    ) == 0
    assert stats["orphan_write_failed"] == 1
    conn.close()


def test_real_sqlite_vec0_dual_write_when_extension_is_available(tmp_path):
    pytest.importorskip("sqlite_vec")
    from core.embeddings import _load_vec

    conn = _orphan_db(tmp_path)
    conn.execute("DROP TABLE vec_embeddings")
    _load_vec(conn)
    conn.execute(
        "CREATE VIRTUAL TABLE vec_embeddings USING vec0("
        "node_id text primary key, embedding float[4] distance_metric=cosine)"
    )
    conn.commit()
    conn.close()


def _real_vec_db(tmp_path: Path, name: str):
    pytest.importorskip("sqlite_vec")
    from core.embeddings import _load_vec
    conn = _orphan_db(tmp_path)
    conn.execute("DROP TABLE vec_embeddings")
    _load_vec(conn)
    conn.execute(
        "CREATE VIRTUAL TABLE vec_embeddings USING vec0("
        "node_id text primary key, embedding float[4] distance_metric=cosine)"
    )
    conn.commit()
    conn.close()
    return sqlite3.connect(str(tmp_path / "orphans.db"))


def test_vec_enable_failure_is_unavailable(tmp_path):
    conn = _real_vec_db(tmp_path, "enable-failure")

    class Disabled(sqlite3.Connection):
        def enable_load_extension(self, _enabled):
            raise sqlite3.OperationalError("extension disabled")

    conn.close()
    disabled = sqlite3.connect(str(tmp_path / "orphans.db"), factory=Disabled)
    assert _vec_write_capability(disabled) is False
    disabled.close()


def test_vec_post_load_query_failure_is_hard(tmp_path):
    _real_vec_db(tmp_path, "query-failure").close()

    class BrokenQuery(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if sql == "SELECT count(*) FROM vec_embeddings":
                raise sqlite3.OperationalError("corrupt vec schema")
            return super().execute(sql, *args, **kwargs)

    conn = sqlite3.connect(str(tmp_path / "orphans.db"), factory=BrokenQuery)
    with pytest.raises(sqlite3.OperationalError, match="corrupt vec schema"):
        _vec_write_capability(conn)
    conn.close()
    conn = sqlite3.connect(str(tmp_path / "orphans.db"))
    client = _Client(lambda n: np.ones((n, 4), dtype=np.float32))
    assert (
        _embed_orphans(
            conn,
            embedding_client=client,
            embedding_model="m",
            expected_dimension=4,
        )
        == 1
    )
    assert conn.execute("SELECT count(*) FROM vec_embeddings").fetchone()[0] == 1
    conn.close()


def test_public_cycle_reports_committed_prefix_when_later_phase_fails(tmp_path, monkeypatch):
    path = _cycle_db(tmp_path)
    monkeypatch.setattr(
        sleep_module, "_find_pairs",
        lambda *_args, **_kwargs: (
            np.asarray([[0, 1]]), np.empty((0, 2), dtype=int), np.eye(2)
        ),
    )
    monkeypatch.setattr(sleep_module, "_compute_metrics", lambda *_args: (_ for _ in ()).throw(RuntimeError("later")))
    result = run_sleep_cycle(db_path=str(path), journal_policy="preserve")
    assert result["status"] == "partial"
    assert result["cross_links_created"] == 1
    check = sqlite3.connect(str(path))
    assert check.execute("SELECT count(*) FROM derivation_edges").fetchone()[0] == 2
    check.close()


def test_early_orphan_commit_survives_candidate_discovery_failure(tmp_path, monkeypatch):
    conn = _orphan_db(tmp_path)
    vector = np.ones(384, dtype=np.float32).tobytes()
    conn.execute(
        "INSERT INTO embeddings VALUES ('anchor', ?, 'all-MiniLM-L6-v2', datetime('now'))",
        (vector,),
    )
    conn.execute("INSERT INTO thought_nodes(id, content) VALUES ('anchor', 'anchor')")
    conn.commit()
    conn.close()
    monkeypatch.setattr(
        sleep_module, "_find_pairs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("discovery")),
    )
    result = run_sleep_cycle(
        db_path=str(tmp_path / "orphans.db"),
        embedding_client=_Client(lambda n: np.ones((n, 384), dtype=np.float32)),
        embedding_model="all-MiniLM-L6-v2",
        expected_dimension=384,
        journal_policy="preserve",
    )
    assert result["status"] == "partial"
    assert result["error"] == "sleep_cycle_failed"
    assert result["orphans_embedded"] == 2
    assert set(result) == set(_empty_sleep_result("partial", "sleep_cycle_failed"))
    check = sqlite3.connect(str(tmp_path / "orphans.db"))
    assert check.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 2
    check.close()


def test_late_orphan_failure_retains_committed_dream_fields(tmp_path, monkeypatch):
    path = _cycle_db(tmp_path, "dream-orphan-failure.db")
    conn = sqlite3.connect(str(path))
    v1 = np.zeros(1024, dtype=np.float32)
    v1[0] = 1.0
    v2 = np.zeros(1024, dtype=np.float32)
    v2[0] = 0.92
    v2[1] = np.sqrt(1.0 - 0.92 ** 2)
    conn.execute("UPDATE thought_nodes SET source_file='one.md' WHERE id='a'")
    conn.execute("UPDATE thought_nodes SET source_file='two.md' WHERE id='b'")
    conn.execute("INSERT INTO thought_nodes(id, content) VALUES ('orphan', 'late orphan')")
    conn.execute("UPDATE embeddings SET vector=? WHERE node_id='a'", (v1.tobytes(),))
    conn.execute("UPDATE embeddings SET vector=? WHERE node_id='b'", (v2.tobytes(),))
    conn.commit()
    conn.close()
    monkeypatch.setattr(
        sleep_module, "_embed_orphans",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("orphan")),
    )
    result = run_sleep_cycle(
        db_path=str(path),
        model_fn=lambda _prompt: "A durable relationship connects these observations.",
        journal_policy="preserve",
    )
    assert result["status"] == "partial"
    assert result["dream_generation"] == "ran"
    assert result["dream_id"]


def _force_late_dream_failure(monkeypatch):
    monkeypatch.setattr(
        sleep_module, "_find_pairs",
        lambda *_args, **_kwargs: (
            np.asarray([[0, 1]]), np.empty((0, 2), dtype=int), np.eye(2)
        ),
    )
    monkeypatch.setattr(
        sleep_module, "_generate_dream",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("late")),
    )


def test_late_failure_preserves_permanence_promotion(tmp_path, monkeypatch):
    path = _cycle_db(tmp_path, "permanence-late.db")

    def promote(conn, **_kwargs):
        conn.execute("UPDATE thought_nodes SET permanent=1 WHERE id='a'")
        conn.commit()
        return {"nodes_promoted": 1}

    monkeypatch.setattr(sleep_module, "_evaluate_permanence", promote)
    monkeypatch.setattr(sleep_module, "_promote_core_memories", lambda *_args: {"promoted": 0, "demoted": 0})
    _force_late_dream_failure(monkeypatch)
    result = run_sleep_cycle(db_path=str(path), model_fn=lambda _prompt: "unused", journal_policy="preserve")
    assert result["status"] == "partial"
    assert result["nodes_made_permanent"] == 1


def test_late_failure_preserves_core_promotion(tmp_path, monkeypatch):
    path = _cycle_db(tmp_path, "core-promotion-late.db")

    def promote_core(conn, _metrics):
        conn.execute("UPDATE thought_nodes SET node_type='core_memory' WHERE id='a'")
        conn.commit()
        return {"promoted": 1, "demoted": 0}

    monkeypatch.setattr(sleep_module, "_evaluate_permanence", lambda *_args: {"nodes_promoted": 0})
    monkeypatch.setattr(sleep_module, "_promote_core_memories", promote_core)
    _force_late_dream_failure(monkeypatch)
    result = run_sleep_cycle(db_path=str(path), model_fn=lambda _prompt: "unused", journal_policy="preserve")
    assert result["status"] == "partial"
    assert result["core_promoted"] == 1


def test_late_failure_preserves_core_demotion(tmp_path, monkeypatch):
    path = _cycle_db(tmp_path, "core-demotion-late.db")
    conn = sqlite3.connect(str(path))
    conn.execute("UPDATE thought_nodes SET node_type='core_memory' WHERE id='a'")
    conn.commit()
    conn.close()

    def demote_core(conn, _metrics):
        conn.execute("UPDATE thought_nodes SET node_type='derived' WHERE id='a'")
        conn.commit()
        return {"promoted": 0, "demoted": 1}

    monkeypatch.setattr(sleep_module, "_evaluate_permanence", lambda *_args: {"nodes_promoted": 0})
    monkeypatch.setattr(sleep_module, "_promote_core_memories", demote_core)
    _force_late_dream_failure(monkeypatch)
    result = run_sleep_cycle(db_path=str(path), model_fn=lambda _prompt: "unused", journal_policy="preserve")
    assert result["status"] == "partial"
    assert result["core_demoted"] == 1


def test_public_cycle_exposes_cross_link_failure(tmp_path, monkeypatch):
    path = _cycle_db(tmp_path)
    monkeypatch.setattr(
        sleep_module, "_find_pairs",
        lambda *_args, **_kwargs: (
            np.asarray([[0, 1]]), np.empty((0, 2), dtype=int), np.eye(2)
        ),
    )
    monkeypatch.setattr(
        sleep_module, "_batch_cross_links",
        lambda *_args, **_kwargs: {"created": 0, "repaired": 0, "skipped": 0,
                                   "failed": 1, "directed_rows": 0},
    )
    result = run_sleep_cycle(db_path=str(path), journal_policy="preserve")
    assert result["status"] in {"failed", "partial"}
    assert result["error"] == "cross_link_failed"


def test_zero_prefix_dream_failure_is_failed(tmp_path, monkeypatch):
    path = _cycle_db(tmp_path, "zero-dream.db")
    conn = sqlite3.connect(str(path))
    conn.execute("UPDATE thought_nodes SET source_file='one.md' WHERE id='a'")
    conn.execute("UPDATE thought_nodes SET source_file='two.md' WHERE id='b'")
    conn.commit()
    conn.close()
    monkeypatch.setattr(
        sleep_module, "_find_pairs",
        lambda *_args, **_kwargs: (
            np.asarray([[0, 1]]), np.empty((0, 2), dtype=int), np.eye(2)
        ),
    )
    monkeypatch.setattr(
        sleep_module, "_batch_cross_links",
        lambda *_args, **_kwargs: {"created": 0, "repaired": 0, "skipped": 0,
                                   "failed": 0, "directed_rows": 0},
    )
    monkeypatch.setattr(sleep_module, "_evaluate_permanence", lambda *_args: {"nodes_promoted": 0})
    monkeypatch.setattr(sleep_module, "_promote_core_memories", lambda *_args: {"promoted": 0, "demoted": 0})
    result = run_sleep_cycle(
        db_path=str(path), model_fn=lambda _prompt: "too short", journal_policy="preserve"
    )
    assert result["dream_generation"] == "failed"
    assert result["status"] == "failed"
    assert result["error"] == "dream_failed"


def test_synchronous_dream_reports_ran(tmp_path, monkeypatch):
    path = _cycle_db(tmp_path, "dream.db")
    conn = sqlite3.connect(str(path))
    v1 = np.zeros(1024, dtype=np.float32)
    v1[0] = 1.0
    v2 = np.zeros(1024, dtype=np.float32)
    v2[0] = 0.92
    v2[1] = np.sqrt(1.0 - 0.92 ** 2)
    conn.execute("UPDATE thought_nodes SET source_file='one.md' WHERE id='a'")
    conn.execute("UPDATE thought_nodes SET source_file='two.md' WHERE id='b'")
    conn.execute("UPDATE embeddings SET vector=? WHERE node_id='a'", (v1.tobytes(),))
    conn.execute("UPDATE embeddings SET vector=? WHERE node_id='b'", (v2.tobytes(),))
    conn.commit()
    conn.close()
    result = run_sleep_cycle(
        db_path=str(path),
        model_fn=lambda _prompt: "A durable relationship connects these two observations.",
        journal_policy="preserve",
    )
    assert result["dream_generation"] == "ran"
    assert result["dream_id"]
    check = sqlite3.connect(str(path))
    assert check.execute(
        "SELECT count(*) FROM thought_nodes WHERE id=?", (result["dream_id"],)
    ).fetchone()[0] == 1
    assert check.execute(
        "SELECT count(*) FROM derivation_edges WHERE child_id=?", (result["dream_id"],)
    ).fetchone()[0] == 2
    check.close()


def test_public_cycle_repairs_orphan_before_anchor_requirement(tmp_path):
    conn = _orphan_db(tmp_path)
    dimension = 384
    anchor = np.ones(dimension, dtype=np.float32).tobytes()
    conn.execute(
        "INSERT INTO embeddings VALUES "
        "('anchor', ?, 'all-MiniLM-L6-v2', datetime('now'))",
        (anchor,),
    )
    conn.execute(
        "INSERT INTO thought_nodes (id, content, decayed) VALUES ('anchor','anchor',0)"
    )
    conn.commit()
    conn.close()
    client = _Client(lambda n: np.ones((n, dimension), dtype=np.float32))
    result = run_sleep_cycle(
        db_path=str(tmp_path / "orphans.db"),
        embedding_client=client,
        embedding_model="all-MiniLM-L6-v2",
        expected_dimension=dimension,
        journal_policy="preserve",
    )
    assert client.calls == [["orphan"]]
    assert result["orphans_embedded"] == 2
    assert result["nodes_selected"] == 2
    assert result["dedup_nodes_merged"] == 1
    check = sqlite3.connect(str(tmp_path / "orphans.db"))
    assert check.execute("SELECT count(*) FROM vec_embeddings").fetchone()[0] == 1
    check.close()


def test_public_cycle_reports_ordinary_only_orphan_write(tmp_path):
    conn = _orphan_db(tmp_path, vec=False)
    conn.close()
    client = _Client(lambda n: np.ones((n, 384), dtype=np.float32))
    result = run_sleep_cycle(
        db_path=str(tmp_path / "orphans.db"),
        embedding_client=client,
        embedding_model="all-MiniLM-L6-v2",
        expected_dimension=384,
        journal_policy="preserve",
    )
    assert result["status"] == "partial"
    assert result["error"] == "vec_capability_unavailable"
    assert result["orphan_vec_unavailable"] == 1
    check = sqlite3.connect(str(tmp_path / "orphans.db"))
    assert check.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 1
    check.close()


def test_decay_audit_schema_is_idempotent_and_preserves_graph(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "audit.db"))
    conn.execute(
        "CREATE TABLE thought_nodes("
        "id TEXT PRIMARY KEY, content TEXT, decayed INTEGER DEFAULT 0, "
        "timestamp TEXT DEFAULT '')"
    )
    conn.execute("INSERT INTO thought_nodes (id, content) VALUES ('n','keep')")
    ensure_decay_audit_schema(conn)
    ensure_decay_audit_schema(conn)
    assert conn.execute("SELECT content FROM thought_nodes").fetchone()[0] == "keep"
    assert conn.execute(
        "SELECT name FROM sqlite_master WHERE name='decay_audit'"
    ).fetchone()
    conn.close()


def test_decay_audit_partial_schema_migrates_and_keeps_history(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "partial-audit.db"))
    conn.execute("CREATE TABLE decay_audit(node_id TEXT, decay_reason TEXT)")
    conn.execute("INSERT INTO decay_audit VALUES ('old', 'dedup_loser')")
    conn.commit()
    ensure_decay_audit_schema(conn)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(decay_audit)")}
    assert {"id", "content_summary", "decay_timestamp", "metadata"} <= columns
    assert conn.execute("SELECT node_id FROM decay_audit").fetchone()[0] == "old"
    conn.close()


def test_preserve_journal_policy_does_not_change_mode(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "empty.db"))
    conn.execute(
        "CREATE TABLE thought_nodes("
        "id TEXT PRIMARY KEY, content TEXT, decayed INTEGER DEFAULT 0, "
        "timestamp TEXT DEFAULT '')"
    )
    conn.commit()
    conn.close()
    result = run_sleep_cycle(
        db_path=str(tmp_path / "empty.db"), journal_policy="preserve"
    )
    assert result["status"] == "unavailable"
    check = sqlite3.connect(str(tmp_path / "empty.db"))
    assert check.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
    check.close()


def test_result_contract_no_llm_has_skipped_dream(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "noemb.db"))
    conn.execute(
        "CREATE TABLE thought_nodes("
        "id TEXT PRIMARY KEY, content TEXT, decayed INTEGER DEFAULT 0, "
        "timestamp TEXT DEFAULT '')"
    )
    conn.execute(
        "CREATE TABLE embeddings("
        "node_id TEXT PRIMARY KEY, vector BLOB, model TEXT, updated_at TEXT)"
    )
    conn.commit()
    conn.close()
    result = run_sleep_cycle(
        db_path=str(tmp_path / "noemb.db"), journal_policy="preserve"
    )
    assert result["status"] == "unavailable"
    assert result["error"] == "too_few_embeddings"
