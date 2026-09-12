"""Production-boundary contracts for the upstream sleep adapter."""

from __future__ import annotations

import sqlite3
import inspect
from pathlib import Path

import numpy as np
import pytest

from core.decay_audit import ensure_decay_audit_schema
from core.sleep import _batch_cross_links, _embed_orphans, run_sleep_cycle


def _edge_db(tmp_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(tmp_path / "edges.db"))
    conn.executescript("""
        CREATE TABLE derivation_edges(parent_id TEXT, child_id TEXT,
            weight REAL, reasoning TEXT, PRIMARY KEY(parent_id, child_id));
    """)
    return conn


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
        "timestamp TEXT DEFAULT '')"
    )
    conn.execute(
        "CREATE TABLE embeddings("
        "node_id TEXT PRIMARY KEY, vector BLOB, model TEXT, updated_at TEXT)"
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
    assert _embed_orphans(conn, expected_dimension=4) == 1
    assert conn.execute("SELECT count(*) FROM vec_embeddings").fetchone()[0] == 1
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


def test_real_sqlite_vec0_dual_write_when_extension_is_available(tmp_path):
    sqlite_vec = pytest.importorskip("sqlite_vec")
    from core.embeddings import _load_vec

    conn = _orphan_db(tmp_path)
    conn.execute("DROP TABLE vec_embeddings")
    _load_vec(conn)
    conn.execute(
        "CREATE VIRTUAL TABLE vec_embeddings USING vec0("
        "node_id text primary key, embedding float[4] distance_metric=cosine)"
    )
    conn.commit()
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


def test_public_cycle_repairs_orphan_before_anchor_requirement(tmp_path):
    conn = _orphan_db(tmp_path)
    anchor = np.ones(4, dtype=np.float32).tobytes()
    conn.execute(
        "INSERT INTO embeddings VALUES ('anchor', ?, 'm', datetime('now'))", (anchor,)
    )
    conn.execute(
        "INSERT INTO thought_nodes (id, content, decayed) VALUES ('anchor','anchor',0)"
    )
    conn.commit()
    client = _Client(lambda n: np.ones((n, 4), dtype=np.float32))
    result = run_sleep_cycle(
        db_path=str(tmp_path / "orphans.db"),
        embedding_client=client,
        embedding_model="all-MiniLM-L6-v2",
        expected_dimension=4,
        journal_policy="preserve",
    )
    assert client.calls == [["orphan"]]
    assert result["orphans_embedded"] == 2
    assert conn.execute("SELECT count(*) FROM vec_embeddings").fetchone()[0] == 2
    conn.close()


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
