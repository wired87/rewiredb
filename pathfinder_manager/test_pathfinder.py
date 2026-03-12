import json
import os

import jax.numpy as jnp

from qbrain.core.qbrain_manager import get_qbrain_table_manager
from qbrain.core.pathfinder_manager.pathfinder import PathfinderManager
from qbrain.jax_test.iterator.iterator import build_time_ctlr, Iterator


def _make_dummy_stores():
    # Simple ragged stores: 1 equation, 1 history, 3 time steps
    in_store = [
        [
            [
                [1.0, 0.0],
                [0.5, 0.5],
                [0.0, 1.0],
            ]
        ]
    ]
    out_store = [
        [
            [
                [0.0, 1.0],
                [0.5, 0.5],
                [1.0, 0.0],
            ]
        ]
    ]
    return in_store, out_store


def test_iterator_extract_segment_arrays_roundtrip():
    in_store, out_store = _make_dummy_stores()
    ctlr = build_time_ctlr(in_store, out_store, d_model=2)
    it = Iterator(d_model=2)

    # Construct a segment over the full flattened range
    in_g, out_g = ctlr
    T = in_g.shape[0]
    N_all = in_g.shape[1] + out_g.shape[1]
    seg = {
        "flat_index_start": 0,
        "flat_index_end": int(T * N_all - 1),
    }

    arrays = it.extract_segment_arrays(ctlr, seg)

    assert "data_states" in arrays
    assert "feature_states" in arrays
    # Basic shape sanity: some entries should be non-empty
    assert arrays["data_states"].size > 0
    assert arrays["feature_states"].size > 0


def test_pathfinder_persist_controller(tmp_path, monkeypatch):
    # Use local DB (DuckDB) by default
    os.environ["LOCAL_DB"] = "True"

    qb = get_qbrain_table_manager()

    # Insert a minimal env row so load_env_run_metadata can find it.
    env_id = "env_test_pathfinder"
    user_id = "user_test_pathfinder"

    qb.set_item(
        "envs",
        items={
            "id": env_id,
            "user_id": user_id,
            "sim_time": 1,
            "cluster_dim": 1,
            "dims": 1,
            "data": json.dumps({}),
            "model_path": str(tmp_path / "dummy_model.json"),
        },
    )

    pathfinder = PathfinderManager(qb=qb)

    # Build a trivial controller directly and persist it.
    in_store, out_store = _make_dummy_stores()
    ctlr, _shape = pathfinder.build_time_controller(in_store, out_store)
    events = pathfinder.identify_recurring_events(ctlr, threshold=0.0)

    cid = pathfinder.persist_time_controller(env_id, user_id, ctlr, events, meta={"test": True})
    assert isinstance(cid, str) and cid

    rows = qb.row_from_id(cid, table="controllers")
    assert rows, "controller row should have been inserted"
    row = rows[0]
    assert row.get("env_id") == env_id
    assert row.get("user_id") == user_id
    assert isinstance(row.get("time_ctlr"), str)


if __name__ == "__main__":
    # Lightweight manual run without pytest
    try:
        test_iterator_extract_segment_arrays_roundtrip()
        test_pathfinder_persist_controller(tmp_path=os.getcwd(), monkeypatch=None)  # type: ignore[arg-type]
        print("Pathfinder tests passed")
    except Exception as e:
        print(f"Pathfinder tests failed: {e}")

