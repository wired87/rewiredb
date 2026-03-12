"""
DB Manager: unified interface for DuckDB (local).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

import duckdb
from duckdb import DuckDBPyConnection

from _db.config import duck_db_path, duck_db_verbose
from _db.log_facade import db_log
from _db.queries import duck_row_from_id

try:
    from _duckdb import DuckDBPyConnection as ddb_res
except ImportError:
    ddb_res = DuckDBPyConnection

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # type: ignore


_db_mgr: Optional["DBManager"] = None


def get_db_manager() -> "DBManager":
    global _db_mgr
    if _db_mgr is None:
        _db_mgr = DBManager()
    return _db_mgr


class DBManager:
    """
    DuckDB manager.
    """

    def __init__(self):
        self._con: DuckDBPyConnection = self._connect()

    def _connect(self) -> DuckDBPyConnection:
        path = duck_db_path()
        directory = os.path.dirname(path)
        if directory:
            # Avoid TOCTOU race when multiple processes initialize in parallel.
            os.makedirs(directory, exist_ok=True)
        try:
            con = duckdb.connect(path)
            db_log("info", "[duck] connect: ok", path=path)
            return con
        except Exception as e:
            msg = str(e).lower()
            if "being used by another process" in msg or "file is already open" in msg:
                # On Windows, another process can hold an exclusive lock.
                # Prefer read-only so we still read canonical data.
                try:
                    con = duckdb.connect(path, read_only=True)
                    db_log("warn", f"DuckDB file locked: {path}. Opened read_only.")
                    return con
                except Exception:
                    pass
                # voita
                # If read_only is also blocked, fall back to a per-process DB file.
                base = Path(path)
                alt = str(base.with_name(f"{base.stem}.{os.getpid()}{base.suffix}"))
                alt_dir = os.path.dirname(alt)
                if alt_dir:
                    os.makedirs(alt_dir, exist_ok=True)
                db_log("warn", f"DuckDB file locked: {path}. Falling back to {alt}")
                return duckdb.connect(alt)
            db_log("error", "[duck] connect: err", path=path, error=str(e))
            raise

    def close(self):
        if self._con:
            self._con.close()
            db_log("info", "[duck] close: ok")
            self._con = None

    def _check(self) -> bool:
        """Quick health check: connection alive and DB reachable."""
        try:
            self._con.execute("SELECT 1").fetchone()
            return True
        except Exception:
            return False

    def _status(self) -> dict:
        """Return DB status: path, tables, connection_alive."""
        tables = []
        try:
            rows = self._con.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main' ORDER BY table_name"
            ).fetchall()
            tables = [r[0] for r in rows]
        except Exception:
            pass
        return {
            "path": duck_db_path(),
            "tables": tables,
            "connection_alive": self._check(),
        }

    def _exec(self, sql: str, params: Optional[List[Any]] = None):
        r: ddb_res = self._con.execute(sql, params) if params is not None else self._con.execute(sql)
        db_log("info", "[duck] exec: ok")
        return r

    def _create_table(self, table_name: str, schema_sql: str) -> None:
        try:
            self._con.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({schema_sql})")
            db_log("info", f"[duck] create_table: ok {table_name}")
        except Exception as e:
            db_log("error", f"[duck] create_table: err {table_name}", error=str(e))

    def _duck_insert(
        self,
        table_name: str,
        rows: list,
        upsert: bool = False,
        conflict_columns: Optional[tuple] = None,
    ) -> bool:
        try:
            if not rows:
                return True
            cols = []
            for row in rows:
                for k in row.keys():
                    if k not in cols:
                        cols.append(k)
            cols_sql = ", ".join(cols)
            placeholders = ", ".join(["?"] * len(cols))

            if upsert and conflict_columns:
                # INSERT OR REPLACE (DuckDB) or ON CONFLICT DO UPDATE
                conflict_cols = ", ".join(conflict_columns)
                set_parts = [f"{c} = excluded.{c}" for c in cols if c not in conflict_columns]
                if set_parts:
                    sql = (
                        f"INSERT INTO {table_name} ({cols_sql}) VALUES ({placeholders}) "
                        f"ON CONFLICT ({conflict_cols}) DO UPDATE SET {', '.join(set_parts)}"
                    )
                else:
                    sql = (
                        f"INSERT OR REPLACE INTO {table_name} ({cols_sql}) VALUES ({placeholders})"
                    )
                for row in rows:
                    vals = [row.get(c) for c in cols]
                    self._con.execute(sql, vals)
            else:
                values = [[row.get(c) for c in cols] for row in rows]
                self._con.executemany(
                    f"INSERT INTO {table_name} ({cols_sql}) VALUES ({placeholders})",
                    values,
                )
            db_log("info", f"[duck] insert: ok {table_name} {len(rows)} rows")
            return True
        except Exception as e:
            db_log("error", f"[duck] insert: err {table_name}", error=str(e))
            return False

    def create_sql_schema(self, schema):
        cols = []
        for k, v in schema.items():
            cols.append(f"{k} {v}")
        schema_sql = ", ".join(cols)
        return schema_sql

    def run_query(
        self,
        sql: str,
        params: Optional[Union[Dict[str, Any], List[Any]]] = None,
        conv_to_dict: bool = False,
    ):
        if params:
            if isinstance(params, dict):
                ordered = []
                for m in re.finditer(r"@(\w+)", sql):
                    k = m.group(1)
                    if k in params:
                        ordered.append(params[k])
                sql = re.sub(r"@\w+", "?", sql)
                cur = self._con.execute(sql, ordered)
            else:
                cur = self._con.execute(sql, list(params))
        else:
            cur = self._con.execute(sql)

        result = cur.fetchall()

        if conv_to_dict and result:
            cols = [d[0] for d in cur.description] if cur.description else []
            return [dict(zip(cols, row)) for row in result]

        return list(result) if result else []

    def execute(
        self,
        sql: str,
        params: Optional[Union[Dict[str, Any], List[Any]]] = None,
    ):
        if isinstance(params, dict):
            ordered = []
            for m in re.finditer(r"@(\w+)", sql):
                k = m.group(1)
                if k in params:
                    ordered.append(params[k])
            sql = re.sub(r"@\w+", "?", sql)
            self._exec(sql, ordered)
        else:
            self._exec(sql, params)

    def insert(
        self,
        table: str,
        rows: Union[Dict, List[Dict]],
        upsert: bool = False,
        conflict_columns: Optional[tuple] = None,
        schema: Optional[Dict[str, str]] = None,
    ) -> bool:
        """
        Insert rows into table. Optionally upsert on conflict.

        Args:
            table: Table name.
            rows: Single dict or list of dicts.
            upsert: If True, replace existing rows on conflict.
            conflict_columns: Columns for conflict detection (default: ("id",)).
            schema: Optional schema for create_if_not_exists; also used for type coercion.
        """
        db_log("info", "insert", table=table, rows=len(rows), upsert=upsert)
        if not isinstance(rows, list):
            rows = [rows]

        if not rows:
            return True

        tbl_schema = self._duck_get_table_schema(
            table, create_if_not_exists=bool(schema), schema=schema
        )

        new_rows = []
        for row in rows:
            new_row = {}

            for col, val in row.items():
                if not isinstance(val, (str, datetime)):
                    if isinstance(val, bytes):
                        val = val.decode("utf-8")
                    val = json.dumps(val)

                new_row[col] = val

                if col not in tbl_schema:
                    self._duck_insert_col(table, col)
            new_rows.append(new_row)

        ok = self._duck_insert(
            table,
            new_rows,
            upsert=upsert,
            conflict_columns=("id",) if upsert and conflict_columns is None else conflict_columns,
        )
        if not ok:
            db_log("error", "insert failed", table=table)
        return ok

    def del_entry(self, nid: str, table: str, user_id: str, name_id: str = "id") -> bool:
        """
        Hard delete entry from DuckDB table.
        """
        try:
            sql = f"""
            DELETE FROM {table}
            WHERE {name_id} = ? AND user_id = ?
            """
            self._con.execute(sql, [nid, user_id])
            return True
        except Exception as e:
            return False

    def create_table(
        self,
        table_name: str,
        schema: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Create table if not exists.

        Args:
            table_name: Table name.
            schema: Optional {column: type} e.g. {"id": "STRING PRIMARY KEY", "name": "STRING"}.
                   If None, uses default: id STRING PRIMARY KEY.
        """
        schema_sql = self.create_sql_schema(schema) if schema else "id STRING PRIMARY KEY"
        try:
            self._create_table(table_name, schema_sql)
        except Exception as e:
            db_log("error", f"[duck] create_table: err {table_name}", error=str(e))

    def insert_col(self, table_id: str, column_name: str, column_type: str):
        return self._duck_insert_col(table_id, column_name, column_type)

    def _duck_insert_col(self, table_id: str, column_name: str):
        try:
            r = self._con.execute(
                "SELECT 1 FROM information_schema.columns WHERE table_schema = 'main' AND table_name = ? AND column_name = ?",
                [table_id, column_name],
            ).fetchone()

            if r:
                return

            col_type = "STRING"
            self._con.execute(
                f"ALTER TABLE {table_id} ADD COLUMN {column_name} {col_type}"
            )

        except Exception as e:
            db_log("error", "Err _duck_insert_col", error=str(e), table=table_id)

    def get_table_schema(self, table_name: str) -> Dict[str, str]:
        """
        Return schema from table (column_name -> data_type).
        Returns {} if table does not exist.
        """
        r = self._con.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_schema = 'main' AND table_name = ?",
            [table_name],
        ).fetchone()
        if not r:
            return {}
        rows = self._con.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = 'main' AND table_name = ? ORDER BY ordinal_position",
            [table_name],
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    def _duck_get_table_schema(
        self,
        table_id: str,
        create_if_not_exists: bool = True,
        schema: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:

        schema = schema or {}

        r = self._con.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_schema = 'main' AND table_name = ?",
            [table_id],
        ).fetchone()

        if r:
            rows = self._con.execute(
                "SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = 'main' AND table_name = ?",
                [table_id],
            ).fetchall()

            return {row[0]: row[1] for row in rows}

        if create_if_not_exists and schema:
            col_defs = [f"{col} {dt}" for col, dt in schema.items()]
            self._create_table(table_id, ", ".join(col_defs))
            return schema
        return schema

    def showup(
        self,
        table_name: Optional[str] = None,
        limit: int = 100,
    ) -> None:

        try:
            from rich.console import Console
            from rich.table import Table
        except ImportError:
            db_log("warn", "rich not installed. Run: pip install rich")
            return

        console = Console()

        def _get_tables() -> List[str]:
            rows = self._con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' ORDER BY table_name"
            ).fetchall()
            return [r[0] for r in rows]

        def _render_table(tbl_name: str):
            query = f"SELECT * FROM {tbl_name} LIMIT {limit}"
            try:
                rows = self.run_query(query, conv_to_dict=True)
            except Exception as e:
                console.print(f"[red]Error querying {tbl_name}: {e}[/red]")
                return

            if not rows:
                console.print(f"[dim]{tbl_name}: (empty)[/dim]")
                return

            table = Table(title=tbl_name, show_header=True, header_style="bold cyan")

            for col in rows[0].keys():
                table.add_column(col, overflow="fold", max_width=40)

            for row in rows:
                table.add_row(*[str(row.get(c, ""))[:80] for c in rows[0].keys()])

            console.print(table)

            if len(rows) >= limit:
                console.print(f"[dim]... (limited to {limit} rows)[/dim]\n")

        if table_name:
            _render_table(table_name)
        else:
            tables = _get_tables()

            if not tables:
                console.print("[yellow]No tables found.[/yellow]")
                return

            for t in tables:
                _render_table(t)
                console.print()

    def print_table(self, table_name: str, limit: Optional[int] = 1000) -> None:
        """
        Render all items of a specific table in the rich terminal.
        Uses DuckDB .show() when possible; falls back to showup() on encoding errors.
        """
        limit_clause = f" LIMIT {limit}" if limit else ""
        query = f"SELECT * FROM {table_name}{limit_clause}"
        try:
            self._con.sql(query).show()
        except (UnicodeEncodeError, UnicodeDecodeError):
            self.showup(table_name=table_name, limit=limit or 1000)
        except Exception as e:
            db_log("error", "print_table failed", table=table_name, error=str(e))

    def status(self) -> dict:
        """Return DB status: path, tables, connection_alive."""
        return self._status()

    def check(self) -> bool:
        """Quick health check: connection alive and DB reachable."""
        return self._check()

    def get_state(self) -> dict:
        """Extended state: status + table row counts (for debug)."""
        state = self._status()
        if duck_db_verbose() >= 2:
            counts = {}
            for tbl in state.get("tables", []):
                try:
                    r = self._con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()
                    counts[tbl] = r[0] if r else 0
                except Exception:
                    counts[tbl] = None
            state["row_counts"] = counts
        return state

    # --- Additional workflow methods (from former workflows module) ---

    def query(self, sql: str, params: Optional[List[Any]] = None):
        """Execute query and return rows."""
        rows = self._con.execute(sql, params).fetchall() if params is not None else self._con.execute(sql).fetchall()
        db_log("info", f"[duck] query: ok {len(rows)} rows")
        return rows

    def query_df(self, sql: str):
        """Execute query and return DataFrame."""
        if pd is None:
            raise ImportError("pandas is required for query_df(). Install pandas to use this helper.")
        df = self._con.execute(sql).df()
        db_log("info", f"[duck] query_df: ok {len(df)} rows")
        return df

    def drop_table(self, table_name: str) -> None:
        self._con.execute(f"DROP TABLE IF EXISTS {table_name}")
        db_log("info", f"[duck] drop_table: ok {table_name}")

    def reset_table(
        self,
        table_name: str,
        schema: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Recreate table with schema. Drops and recreates.

        Args:
            table_name: Table name.
            schema: {column: type} e.g. {"id": "STRING PRIMARY KEY", "name": "STRING"}.
                   If None, uses id STRING PRIMARY KEY.
        """
        schema_sql = self.create_sql_schema(schema) if schema else "id STRING PRIMARY KEY"
        self._con.execute(f"CREATE OR REPLACE TABLE {table_name} ({schema_sql})")
        db_log("info", f"[duck] reset_table: ok {table_name}")

    def insert_raw(self, table: str, columns: List[str], values: List[Any]) -> None:
        placeholders = ",".join(["?"] * len(values))
        cols = ",".join(columns)
        self._con.execute(
            f"INSERT INTO {table} ({cols}) VALUES ({placeholders})",
            values,
        )
        db_log("info", f"[duck] db_insert: ok {table} 1 row")

    def replace(
        self,
        table: str,
        rows: Union[Dict, List[Dict]],
        conflict_columns: tuple = ("id",),
        schema: Optional[Dict[str, str]] = None,
    ) -> bool:
        """
        Replace rows (upsert). Uses schema for create_if_not_exists and type coercion.

        Args:
            table: Table name.
            rows: Single dict or list of dicts.
            conflict_columns: Columns for ON CONFLICT (default: id).
            schema: Optional schema for create_if_not_exists; used when table is missing.
        """
        return self.insert(
            table,
            rows=rows,
            upsert=True,
            conflict_columns=conflict_columns,
            schema=schema,
        )

    def update(
        self,
        table: str,
        set_clause: str,
        where_clause: str,
        params: Optional[List[Any]] = None,
    ) -> None:
        """
        Update rows. Params for prepared statement placeholders.
        """
        if params is not None:
            self._con.execute(
                f"UPDATE {table} SET {set_clause} WHERE {where_clause}",
                params,
            )
        else:
            self._con.execute(f"UPDATE {table} SET {set_clause} WHERE {where_clause}")
        db_log("info", f"[duck] update: ok {table}")

    def delete(
        self,
        table: str,
        where_clause: str,
        params: Optional[List[Any]] = None,
    ) -> None:
        """Delete rows. Params for prepared statement placeholders in where_clause."""
        if params is not None:
            self._con.execute(f"DELETE FROM {table} WHERE {where_clause}", params)
        else:
            self._con.execute(f"DELETE FROM {table} WHERE {where_clause}")
        db_log("info", f"[duck] delete: ok {table}")

    def register_df(self, df: "pd.DataFrame", view_name: str) -> None:
        if pd is None:
            raise ImportError("pandas is required for register_df(). Install pandas to use this helper.")
        self._con.register(view_name, df)

    def insert_df(self, table_name: str, df: "pd.DataFrame") -> None:
        if pd is None:
            raise ImportError("pandas is required for insert_df(). Install pandas to use this helper.")
        self._con.register("tmp_df", df)
        self._con.execute(f"INSERT INTO {table_name} SELECT * FROM tmp_df")
        db_log("info", f"[duck] insert_df: ok {table_name} {len(df)} rows")

    def read_csv(self, path: str, table_name: str) -> None:
        self._con.execute(
            f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto('{path}')"
        )
        db_log("info", f"[duck] read_csv: ok {table_name}")

    def read_parquet(self, path: str, table_name: str) -> None:
        self._con.execute(
            f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_parquet('{path}')"
        )
        db_log("info", f"[duck] read_parquet: ok {table_name}")

    def export_csv(self, table_name: str, path: str) -> None:
        self._con.execute(
            f"COPY {table_name} TO '{path}' (HEADER, DELIMITER ',')"
        )
        db_log("info", f"[duck] export_csv: ok {table_name} -> {path}")

    def export_parquet(self, table_name: str, path: str) -> None:
        self._con.execute(
            f"COPY {table_name} TO '{path}' (FORMAT PARQUET)"
        )
        db_log("info", f"[duck] export_parquet: ok {table_name} -> {path}")


    def row_from_id(self, nid:list or str, table, select="*", user_id=None):
        #print("retrieve_env_from_id...", nid)
        if isinstance(nid, str):
            nid = [nid]

        query, params = duck_row_from_id(
            table=table,
            ids=nid,
            select=select,
            user_id=user_id,
        )

        items = self.run_query(query, params=params, conv_to_dict=True)
        print(f"row_from_id fetched items for {nid}: ", len(items))
        return items



    @property
    def connection(self):
        return self._con


def db_check() -> bool:
    """Standalone health check (uses singleton manager)."""
    return get_db_manager().check()


def db_status() -> dict:
    """Standalone status (uses singleton manager)."""
    return get_db_manager().status()

if __name__ == "__main__":
    db = DBManager()
    db.print_table("params")       # Shows up to 1000 rows
    #db.print_table("params", None) # Shows all rows
