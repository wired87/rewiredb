import json
import os
from typing import Any, Dict, List, Optional, Tuple

import jax.numpy as jnp
import pandas as pd

from qbrain.core.qbrain_manager import get_qbrain_table_manager
from qbrain.jax_test.iterator.iterator import Iterator, build_time_ctlr


class PathfinderManager:
    """
    Offline analyser for GTM runs.

    - Loads env/model metadata for a given (env_id, user_id)
    - Reconstructs (in_store, out_store) compatible with build_time_ctlr
    - Builds time controller (in_grid, out_grid)
    - Detects recurring events
    - Persists controller struct into the controllers table
    """

    def __init__(self, g, qb=None, iterator: Optional[Iterator] = None, d_model: int = 64):
        self.qb = qb or get_qbrain_table_manager()
        self.g=g
        self.iterator = iterator or Iterator(d_model=d_model)
        self.d_model = d_model

    # ---- Metadata / engine output loading ----

    def create_edge(self, operator_idx, param_idx, time_id=0):
        return (time_id, operator_idx, param_idx)

    def filter_nodes(self):
        sorted_operators = sorted(
            [v for k, v in self.g.G.nodes(data=True) if v.get('type') == 'OPERATOR'],
            key=lambda x: x.get('op_idx', float('inf'))
        )
        sorted_params = sorted(
            [v for k, v in self.g.G.nodes(data=True) if v.get('type') == 'PARAMS'],
            key=lambda x: x.get('op_idx', float('inf'))
        )
        return sorted_operators, sorted_params

    def create_nodes(self, operators, params):
        from qbrain.embedder import embed
        operators = jnp.stack([embed(v) for v in operators])
        params = jnp.stack([embed(v) for v in params])
        return operators, params


    def main(self, goal="psi"):
        sorted_operators, sorted_params = self.filter_nodes()

        binary_ops, unary_ops = self.bin_unary_op_classification()

        operators, params = self.create_nodes(sorted_operators, sorted_params)

        op_ids, params_ids = self.get_ids(sorted_operators, sorted_params)

        operator_edges, params_edges = self.create_edges(sorted_operators, sorted_params, op_ids, params_ids)

        equation = self.find(
            param_ids=params_ids,
            param_embeddings=params,
            goal=goal,
            binary_operators=binary_ops,
            unary_operators=unary_ops
        )


    def bin_unary_op_classification(self):
        sorted_operators, sorted_params = self.filter_nodes()
        binary_ops = []
        unary_ops = []
        for node in sorted_operators:
            if node.get('sub_type') == 'binary':
                binary_ops.append(node.get('id'))
            else:
                unary_ops.append(node.get('id'))
        return binary_ops, unary_ops


    def find(
            self,
            param_ids,
            param_embeddings:jnp.array,
            goal,
            binary_operators,
            unary_operators,
    ):
        from pysr import PySRRegressor
        model = PySRRegressor(
            niterations=40,  # Anzahl der Such-Zyklen
            # Hier definierst du EXAKT die Operatoren aus deinem Grid-Layer
            binary_operators=binary_operators, #["+", "*", "/"],
            unary_operators=unary_operators, #["square", "inv(x) = 1/x"],

            # Deine Parameter-Namen (Knoten-Labels)
            variable_names=param_ids, #["masse", "geschwindigkeit"],

            # Komplexitäts-Kontrolle (Dein Ansatz mit der Länge!)
            # 'parsimony' bestraft zu lange Gleichungen
            parsimony=0.001,
            maxsize=10,  # Maximale Anzahl an Operatoren + Parametern

            # Verhindert Division durch Null etc.
            constraints={'/': (-1, 1)},
        )
        param_embed_df = pd.DataFrame(param_embeddings)
        equation = model.fit(param_embed_df, goal)
        print("eq identified", equation)
        return equation

    def get_ids(self, sorted_operators, sorted_params):
        op_ids = [node["id"] for node in sorted_operators]
        params_ids = [node["id"] for node in sorted_params]
        return op_ids, params_ids


    def create_edges(self, sorted_operators, sorted_params, op_ids, params_ids):
        operator_edges= []
        params_edges= []

        for node in sorted_operators:
            for src, trgt, attrs in self.g.G.edges(data=True):
                if node["id"] == src and trgt in params_ids:
                    operator_edges.append(
                        [node["id"], trgt]
                    )
        for node in sorted_params:
            for src, trgt, attrs in self.g.G.edges(data=True):
                if node["id"] == src and trgt in op_ids:
                    params_edges.append(
                        [node["id"], trgt]
                    )
        return operator_edges, params_edges






    def load_env_run_metadata(self, env_id: str, user_id: str) -> Dict[str, Any]:
        rows = self.qb.row_from_id(env_id, table="envs", user_id=user_id, select="*")
        if not rows:
            return {}
        row = rows[0]
        return {
            "env": row,
            "model_path": row.get("model_path"),
            "model_data_path": row.get("model_data_path"),
            "pattern": row.get("pattern"),
        }

    def load_engine_stores(
        self, run_meta: Dict[str, Any]
    ) -> Tuple[List[List[List[Any]]], List[List[List[Any]]], Dict[str, Any]]:
        """
        Best‑effort reconstruction of (in_store, out_store) from stored artifacts.

        For now this uses a very lightweight protocol:
        - If model_data_path points to a JSON or NPZ file with keys 'in_store', 'out_store',
          they are loaded directly.
        - Otherwise, fall back to empty ragged stores so build_time_ctlr still returns
          a valid minimal controller.
        """
        in_store: List[List[List[Any]]] = [[[]]]
        out_store: List[List[List[Any]]] = [[[]]]
        aux: Dict[str, Any] = {}

        data_path = run_meta.get("model_data_path") or run_meta.get("model_path")
        if not data_path or not isinstance(data_path, str):
            return in_store, out_store, aux

        data_path = os.path.abspath(data_path)
        if not os.path.isfile(data_path):
            return in_store, out_store, aux

        try:
            if data_path.endswith(".json"):
                with open(data_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                in_store = payload.get("in_store") or in_store
                out_store = payload.get("out_store") or out_store
                aux = {k: v for k, v in payload.items() if k not in ("in_store", "out_store")}
            elif data_path.endswith(".npz"):
                # Deferred heavy decoding – keep aux as a simple marker for now.
                aux = {"npz_path": data_path}
            else:
                aux = {"raw_path": data_path}
        except Exception as e:
            # Keep system robust even when decoding fails.
            aux = {"error": str(e), "raw_path": data_path}

        return in_store, out_store, aux

    # ---- Time controller construction ----

    def build_time_controller(self, in_store, out_store):
        ctlr = build_time_ctlr(in_store, out_store, d_model=self.d_model)
        in_g, out_g = ctlr
        shape = {
            "in_shape": tuple(in_g.shape),
            "out_shape": tuple(out_g.shape),
        }
        return ctlr, shape

    # ---- Event detection on controller ----

    def identify_recurring_events(
        self, ctlr, threshold: float = 0.9
    ) -> List[Dict[str, Any]]:
        """
        Simple recurring‑event detector on top of scan_in_out_features.

        Uses scores per time/feature and groups consecutive high‑score positions
        into coarse segments.
        """
        scores, idx_within_t = self.iterator.scan_in_out_features(ctlr)
        if scores.size == 0:
            return []

        # Normalize scores to [0,1] range for stability.
        max_score = float(jnp.max(scores))
        if max_score > 0:
            norm_scores = scores / max_score
        else:
            norm_scores = scores

        events: List[Dict[str, Any]] = []
        T = int(ctlr[0].shape[0])
        N_all = int(ctlr[0].shape[1] + ctlr[1].shape[1])

        current_start: Optional[int] = None
        for flat_idx in range(norm_scores.shape[0]):
            s = float(norm_scores[flat_idx])
            if s >= threshold:
                if current_start is None:
                    current_start = flat_idx
            else:
                if current_start is not None:
                    seg = self._segment_from_range(
                        ctlr=ctlr,
                        scores=norm_scores,
                        start=current_start,
                        end=flat_idx - 1,
                        T=T,
                        N_all=N_all,
                    )
                    events.append(seg)
                    current_start = None

        if current_start is not None:
            seg = self._segment_from_range(
                ctlr=ctlr,
                scores=norm_scores,
                start=current_start,
                end=norm_scores.shape[0] - 1,
                T=T,
                N_all=N_all,
            )
            events.append(seg)

        return events

    def _segment_from_range(
        self,
        ctlr,
        scores: jnp.ndarray,
        start: int,
        end: int,
        T: int,
        N_all: int,
    ) -> Dict[str, Any]:
        """Helper to map flat index range back to (t, idx_within_t) region."""
        flat_indices = jnp.arange(start, end + 1, dtype=jnp.int32)
        t_indices = (flat_indices // N_all).tolist()
        v_indices = (flat_indices % N_all).tolist()
        seg_scores = scores[start : end + 1]

        return {
            "t_start": int(min(t_indices)) if t_indices else 0,
            "t_end": int(max(t_indices)) if t_indices else 0,
            "score": float(jnp.max(seg_scores)) if seg_scores.size else 0.0,
            "flat_index_start": int(start),
            "flat_index_end": int(end),
            "t_indices": t_indices,
            "var_indices": v_indices,
            "tag": "recurring_feature",
        }

    # ---- Equation inference on recurring events ----

    def infer_equation_for_event(
        self,
        ctlr,
        event_segment: Dict[str, Any],
        *,
        target: str = "feature_states",
    ) -> Dict[str, Any]:
        """
        Infer a simple time-dependent equation that approximates the motion
        of a recurring event in a stable way.

        Currently fits a per-dimension linear model x(t) ≈ a * t + b over the
        selected segment. The result can be used as a compact description of
        the underlying trajectory.
        """
        seg = {
            "flat_index_start": event_segment.get("flat_index_start", 0),
            "flat_index_end": event_segment.get("flat_index_end", 0),
        }
        arrays = self.iterator.extract_segment_arrays(ctlr, seg)
        series = arrays.get(target)

        if series is None or series.size == 0:
            return {
                "type": "none",
                "equation": "",
                "coefficients": None,
            }

        # Flatten over samples; treat each row as observation over implicit time t = 0..N-1
        series = jnp.asarray(series)
        N = series.shape[0]
        t = jnp.arange(N, dtype=series.dtype)
        # Design matrix [t, 1]
        X = jnp.stack([t, jnp.ones_like(t)], axis=-1)  # (N,2)

        # Solve least squares for each dimension independently
        y = jnp.reshape(series, (N, -1))  # (N,D)
        XtX = X.T @ X
        XtX_inv = jnp.linalg.pinv(XtX)
        beta = XtX_inv @ (X.T @ y)  # (2,D)

        a = beta[0]  # slope per dimension
        b = beta[1]  # intercept per dimension

        eq_str = "x(t) = a * t + b"
        return {
            "type": "linear_time",
            "equation": eq_str,
            "coefficients": {
                "a": a.tolist(),
                "b": b.tolist(),
            },
            "target": target,
        }

    # ---- Persistence layer ----

    def persist_time_controller(
        self,
        env_id: str,
        user_id: str,
        ctlr,
        event_segments: List[Dict[str, Any]],
        meta: Optional[Dict[str, Any]] = None,
        controller_id: Optional[str] = None,
    ) -> str:
        """
        Serialize and upsert time controller into controllers table.
        """
        in_g, out_g = ctlr
        time_ctlr_struct = {
            "env_id": env_id,
            "user_id": user_id,
            "ctlr_shape": {
                "in_shape": tuple(in_g.shape),
                "out_shape": tuple(out_g.shape),
            },
            "event_segments": event_segments,
            "meta": meta or {},
        }

        # id: allow external ID generation, else simple env-scoped key.
        cid = controller_id or f"{env_id}__{user_id}"

        item = {
            "id": cid,
            "env_id": env_id,
            "user_id": user_id,
            "time_ctlr": json.dumps(time_ctlr_struct),
            "event_type": "recurring_feature",
            "event_signature": f"{env_id}:{len(event_segments)}",
            "meta": meta or {},
        }

        self.qb.set_item(
            "controllers",
            items=item,
            keys={"id": cid, "env_id": env_id, "user_id": user_id},
        )
        return cid

    # ---- High-level one-shot API ----

    def build_and_persist_for_env(
        self,
        env_id: str,
        user_id: str,
        *,
        threshold: float = 0.9,
    ) -> Optional[str]:
        """
        Convenience entry: load metadata, build controller, detect events, and persist.
        Returns controller_id or None when nothing useful could be built.
        """
        meta = self.load_env_run_metadata(env_id, user_id)
        if not meta:
            return None

        in_store, out_store, aux = self.load_engine_stores(meta)
        ctlr, _shape = self.build_time_controller(in_store, out_store)
        events = self.identify_recurring_events(ctlr, threshold=threshold)
        controller_meta = {"source": "post_run"}
        if aux:
            controller_meta["aux"] = aux
        return self.persist_time_controller(env_id, user_id, ctlr, events, meta=controller_meta)

