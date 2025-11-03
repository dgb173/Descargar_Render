"""
Modulo de filtrado rapido de patrones por handicaps.

Incluye:
    - Pre-procesado vectorizado de dataframes de partidos.
    - Filtros progresivos (local -> visitante) con tolerancia +-0.25.
    - Calculo ponderado de similitud con pesos configurados.
    - Busqueda incremental con streaming opcional y devolucion ordenada.
    - Modo CLI: ``python -m modules.pattern_filter_fast --seed-id XXX --top-k 30``.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional

import numpy as np
import pandas as pd

try:
    from .utils import parse_ah_to_number_of, format_ah_as_decimal_string_of
except ImportError:  # pragma: no cover - ejecutado cuando no esta disponible el paquete
    # Uso local (por ejemplo, si se ejecuta el modulo de forma aislada fuera del paquete).
    from modules.utils import parse_ah_to_number_of, format_ah_as_decimal_string_of  # type: ignore


# Pesos definidos por el jefe.
HANDICAP_WEIGHT = 40.0
LINE_MOVE_WEIGHT = 20.0
FAVORITE_WEIGHT = 20.0
RECENT_FORM_WEIGHT = 10.0
OTHER_WEIGHT = 10.0

# Config general.
HANDICAP_TOLERANCE = 0.25
GOAL_LINE_SCALE = 1.5  # escala para comparar lineas de goles

# Tipos.
CallbackType = Optional[Callable[[Dict[str, object]], None]]


def _coerce_float(value: object) -> float:
    """Convierte a float cuando sea posible, devolviendo NaN en caso contrario."""
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        txt = value.strip()
        if not txt:
            return np.nan
        try:
            return float(txt)
        except ValueError:
            parsed = parse_ah_to_number_of(txt)
            return float(parsed) if parsed is not None else np.nan
    if isinstance(value, dict):
        return _coerce_float(value.get("value"))
    return np.nan


def _safe_team_key(name: object) -> Optional[str]:
    if not isinstance(name, str) or not name.strip():
        return None
    return name.strip().lower()


def _prepare_match_id(series: pd.Series) -> pd.Series:
    """Asegura que la columna match_id exista y sea string."""
    if series.name == "match_id":
        return series.astype(str)
    return series.fillna("").astype(str)


class PatternFilter:
    """Filtro inteligente de patrones basado en handicaps asiaticos."""

    def __init__(self, cache_df: pd.DataFrame, upcoming_df: pd.DataFrame) -> None:
        if cache_df is None or upcoming_df is None:
            raise ValueError("Se requieren dataframes de cache y de proximos partidos.")

        self.cache_df = self._prepare_dataframe(cache_df.copy(), is_cache=True)
        self.upcoming_df = self._prepare_dataframe(upcoming_df.copy(), is_cache=False)

        # Indice rapido por match_id para el cache.
        self.cache_df = (
            self.cache_df.drop_duplicates(subset="match_id")
            .set_index("match_id", drop=False)
            .sort_index()
        )

        # Lookup de forma reciente en base a partidos con resultado.
        self._team_form = self._build_recent_form_lookup(self.cache_df)

        # Resultados de la ultima busqueda (para CLI/utilidades).
        self._last_results: List[Dict[str, object]] = []

    # --------------------------------------------------------------------- #
    # Pre-procesado
    # --------------------------------------------------------------------- #
    def _prepare_dataframe(self, df: pd.DataFrame, *, is_cache: bool) -> pd.DataFrame:
        if df.empty:
            return df.assign(
                match_id=pd.Series(dtype=str),
                ah_local=pd.Series(dtype=float),
                ah_away=pd.Series(dtype=float),
                line_move_ah_value=pd.Series(dtype=float),
                goal_line_value=pd.Series(dtype=float),
                home_is_favorite=pd.Series(dtype=float),
                start_time=pd.Series(dtype="datetime64[ns]"),
            )

        # Asegurar columna match_id.
        if "match_id" not in df.columns and "id" in df.columns:
            df["match_id"] = df["id"]
        if "match_id" not in df.columns:
            raise ValueError("No se encontro columna 'match_id' ni 'id' en el dataframe.")
        df["match_id"] = _prepare_match_id(df["match_id"])

        # Normalizar columnas de hora.
        df["start_time"] = self._normalize_start_time(df)

        # Handicaps local/visitante.
        local, away = self._extract_handicap_columns(df)
        df["ah_local"] = local
        df["ah_away"] = away

        # Movimiento de linea (si existe).
        df["line_move_ah_value"] = self._extract_line_move(df)

        # Goal line generica (otros factores).
        df["goal_line_value"] = self._extract_goal_line(df)

        # Favorito local segun signo del handicap.
        df["home_is_favorite"] = np.where(
            df["ah_local"].isna() | np.isclose(df["ah_local"], 0.0, atol=1e-8),
            np.nan,
            df["ah_local"] < 0,
        )

        if is_cache:
            # Asegurar columnas base para analisis posteriores.
            for col in ("home_team", "away_team"):
                if col not in df.columns:
                    df[col] = np.nan

        return df

    def _normalize_start_time(self, df: pd.DataFrame) -> pd.Series:
        series = pd.Series([pd.NaT] * len(df), index=df.index, dtype="datetime64[ns]")
        datetime_cols = ["start_time", "time_obj", "kickoff", "date"]

        for col in datetime_cols:
            if col in df.columns:
                parsed = pd.to_datetime(df[col], errors="coerce", utc=False)
                series = series.fillna(parsed)

        # Fallback: intentar generar datetime desde fecha/hora divididas.
        if series.isna().all():
            if {"match_date", "time"}.issubset(df.columns):
                combined = df["match_date"].fillna("") + " " + df["time"].fillna("")
                series = pd.to_datetime(combined, errors="coerce", utc=False)

        return series

    def _extract_handicap_columns(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        series_local = pd.Series(np.nan, index=df.index, dtype=float)
        series_away = pd.Series(np.nan, index=df.index, dtype=float)

        # Caso: columna ah_current con diccionarios.
        if "ah_current" in df.columns:
            mask_dict = df["ah_current"].apply(lambda v: isinstance(v, dict))
            if mask_dict.any():
                extracted = (
                    pd.json_normalize(df.loc[mask_dict, "ah_current"])
                    .rename(columns=str.lower)
                    .reindex(columns=["local", "home", "away", "visitante"])
                )
                if "local" in extracted:
                    series_local.loc[mask_dict] = pd.to_numeric(
                        extracted["local"], errors="coerce"
                    )
                elif "home" in extracted:
                    series_local.loc[mask_dict] = pd.to_numeric(
                        extracted["home"], errors="coerce"
                    )
                if "away" in extracted:
                    series_away.loc[mask_dict] = pd.to_numeric(
                        extracted["away"], errors="coerce"
                    )
                elif "visitante" in extracted:
                    series_away.loc[mask_dict] = pd.to_numeric(
                        extracted["visitante"], errors="coerce"
                    )

        # Columnas explicitas.
        for col in (
            "ah_local",
            "ah_home",
            "handicap_local",
            "home_handicap",
            "home_ah",
        ):
            if col in df.columns:
                series_local = pd.to_numeric(df[col], errors="coerce")
                break

        for col in (
            "ah_away",
            "handicap_away",
            "away_handicap",
            "away_ah",
        ):
            if col in df.columns:
                series_away = pd.to_numeric(df[col], errors="coerce")
                break

        # Fallback: unica columna "handicap".
        if series_local.isna().all() and "handicap" in df.columns:
            parsed = df["handicap"].apply(parse_ah_to_number_of)
            series_local = parsed.apply(lambda v: np.nan if v is None else float(v))

        if series_away.isna().all():
            series_away = -series_local

        return series_local.astype(float), series_away.astype(float)

    def _extract_line_move(self, df: pd.DataFrame) -> pd.Series:
        for col in ("line_move_ah", "line_move", "ah_move"):
            if col in df.columns:
                return df[col].apply(_coerce_float)
        return pd.Series(np.nan, index=df.index, dtype=float)

    def _extract_goal_line(self, df: pd.DataFrame) -> pd.Series:
        for col in ("goal_line", "total_line", "goles_linea"):
            if col in df.columns:
                return df[col].apply(_coerce_float)
        return pd.Series(np.nan, index=df.index, dtype=float)

    def _build_recent_form_lookup(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        if df.empty or "score" not in df.columns:
            return {}

        history: Dict[str, Dict[str, float]] = {}

        for _, row in df.iterrows():
            score = row.get("score")
            home = _safe_team_key(row.get("home_team"))
            away = _safe_team_key(row.get("away_team"))

            if not score or home is None or away is None:
                continue

            try:
                parts = score.replace(":", "-").split("-")
                home_goals, away_goals = int(parts[0]), int(parts[1])
            except (ValueError, IndexError, AttributeError):
                continue

            if home_goals > away_goals:
                home_result, away_result = 1.0, 0.0
            elif home_goals < away_goals:
                home_result, away_result = 0.0, 1.0
            else:
                home_result = away_result = 0.5  # empate

            for team, result in ((home, home_result), (away, away_result)):
                stats = history.setdefault(team, {"total": 0.0, "sum": 0.0})
                stats["total"] += 1.0
                stats["sum"] += result

        # convertimos a promedio 0-1
        for stats in history.values():
            if stats["total"] > 0:
                stats["form_score"] = stats["sum"] / stats["total"]
            else:
                stats["form_score"] = 0.5

        return history

    # --------------------------------------------------------------------- #
    # Logica de filtros
    # --------------------------------------------------------------------- #
    def _get_seed_row(self, seed_id: str) -> Optional[pd.Series]:
        try:
            return self.cache_df.loc[str(seed_id)]
        except KeyError:
            return None

    def _vector_handicap_mask(self, series: pd.Series, seed_value: float) -> pd.Series:
        if pd.isna(seed_value):
            return pd.Series(False, index=series.index)

        tolerance = HANDICAP_TOLERANCE + 1e-9
        series = series.astype(float)

        same_sign = np.sign(series.fillna(seed_value)) == np.sign(seed_value)

        # caso especial: ambos ~0
        near_zero = np.isclose(series, 0.0, atol=1e-6) & np.isclose(seed_value, 0.0, atol=1e-6)
        diff_ok = (series - seed_value).abs() <= tolerance

        return (same_sign | near_zero) & diff_ok

    def _filter_candidates(self, seed_row: pd.Series) -> pd.DataFrame:
        if self.upcoming_df.empty:
            return self.upcoming_df.iloc[0:0]

        seed_local = seed_row.get("ah_local")
        seed_away = seed_row.get("ah_away")

        candidates = self.upcoming_df[self.upcoming_df["match_id"] != seed_row["match_id"]]

        if candidates.empty or pd.isna(seed_local) or pd.isna(seed_away):
            return candidates.iloc[0:0]

        mask_local = self._vector_handicap_mask(candidates["ah_local"], seed_local)
        filtered = candidates.loc[mask_local]

        if filtered.empty:
            return filtered

        mask_away = self._vector_handicap_mask(filtered["ah_away"], seed_away)
        return filtered.loc[mask_away]

    # --------------------------------------------------------------------- #
    # Calculo de similitud
    # --------------------------------------------------------------------- #
    def _handicap_component(self, seed_value: float, cand_value: float) -> float:
        if pd.isna(seed_value) or pd.isna(cand_value):
            return 0.0
        if not self._similar_handicap(seed_value, cand_value):
            return 0.0
        diff = abs(seed_value - cand_value)
        return max(0.0, 1.0 - diff / HANDICAP_TOLERANCE)

    def _line_move_component(self, seed_value: float, cand_value: float) -> float:
        if pd.isna(seed_value) and pd.isna(cand_value):
            return 0.0
        if pd.isna(seed_value) or pd.isna(cand_value):
            return 0.2  # ligera coincidencia si uno carece de dato
        diff = abs(seed_value - cand_value)
        return max(0.0, 1.0 - diff / HANDICAP_TOLERANCE)

    def _favorite_component(self, seed_flag: object, cand_flag: object) -> float:
        if pd.isna(seed_flag) and pd.isna(cand_flag):
            return 0.5
        if pd.isna(seed_flag) or pd.isna(cand_flag):
            return 0.25
        return 1.0 if bool(seed_flag) == bool(cand_flag) else 0.0

    def _recent_form_component(self, seed_row: pd.Series, cand_row: pd.Series) -> float:
        def form_delta(row: pd.Series) -> Optional[float]:
            home_key = _safe_team_key(row.get("home_team"))
            away_key = _safe_team_key(row.get("away_team"))
            if home_key is None or away_key is None:
                return None
            home_form = self._team_form.get(home_key, {}).get("form_score")
            away_form = self._team_form.get(away_key, {}).get("form_score")
            if home_form is None or away_form is None:
                return None
            return home_form - away_form

        seed_delta = form_delta(seed_row)
        cand_delta = form_delta(cand_row)

        if seed_delta is None or cand_delta is None:
            return 0.0

        diff = abs(seed_delta - cand_delta)
        return max(0.0, 1.0 - min(diff, 1.0))

    def _other_component(self, seed_row: pd.Series, cand_row: pd.Series) -> float:
        seed_goal = seed_row.get("goal_line_value")
        cand_goal = cand_row.get("goal_line_value")

        if pd.isna(seed_goal) or pd.isna(cand_goal):
            return 0.0

        diff = abs(seed_goal - cand_goal)
        return max(0.0, 1.0 - min(diff / GOAL_LINE_SCALE, 1.0))

    def _similar_handicap(self, seed_value: float, cand_value: float) -> bool:
        if pd.isna(seed_value) or pd.isna(cand_value):
            return False
        seed_sign = np.sign(seed_value)
        cand_sign = np.sign(cand_value)
        if np.isclose(seed_value, 0.0, atol=1e-6) and np.isclose(cand_value, 0.0, atol=1e-6):
            sign_ok = True
        else:
            sign_ok = seed_sign == cand_sign
        return sign_ok and abs(seed_value - cand_value) <= HANDICAP_TOLERANCE

    def compute_similarity(self, seed_row: pd.Series, cand_row: pd.Series) -> float:
        seed_local = seed_row.get("ah_local")
        seed_away = seed_row.get("ah_away")
        cand_local = cand_row.get("ah_local")
        cand_away = cand_row.get("ah_away")

        handicap_local = self._handicap_component(seed_local, cand_local)
        handicap_away = self._handicap_component(seed_away, cand_away)
        handicap_score = (handicap_local + handicap_away) / 2.0

        line_move_score = self._line_move_component(
            seed_row.get("line_move_ah_value"), cand_row.get("line_move_ah_value")
        )
        favorite_score = self._favorite_component(
            seed_row.get("home_is_favorite"), cand_row.get("home_is_favorite")
        )
        form_score = self._recent_form_component(seed_row, cand_row)
        other_score = self._other_component(seed_row, cand_row)

        total = (
            handicap_score * HANDICAP_WEIGHT
            + line_move_score * LINE_MOVE_WEIGHT
            + favorite_score * FAVORITE_WEIGHT
            + form_score * RECENT_FORM_WEIGHT
            + other_score * OTHER_WEIGHT
        )

        return round(min(total, 100.0), 2)

    # --------------------------------------------------------------------- #
    # Busqueda incremental
    # --------------------------------------------------------------------- #
    def incremental_search(
        self,
        seed_id: str,
        *,
        top_k: Optional[int] = None,
        callback: CallbackType = None,
    ) -> Iterator[Dict[str, object]]:
        """Generador que emite resultados parciales y un resumen final."""
        seed_row = self._get_seed_row(seed_id)
        if seed_row is None:
            payload = {
                "status": "error",
                "seed_id": str(seed_id),
                "message": f"No se encontro el partido semilla '{seed_id}' en el cache.",
            }
            if callback:
                callback(payload)
            yield payload
            self._last_results = []
            return

        filtered = self._filter_candidates(seed_row)
        if filtered.empty:
            summary = {
                "status": "complete",
                "seed_id": str(seed_id),
                "count": 0,
                "results": [],
                "message": "No se encontraron partidos similares.",
            }
            if callback:
                callback(summary)
            yield summary
            self._last_results = []
            return

        collected: List[Dict[str, object]] = []

        for _, cand_row in filtered.iterrows():
            similarity = self.compute_similarity(seed_row, cand_row)

            result = {
                "match_id": cand_row.get("match_id"),
                "home_team": cand_row.get("home_team"),
                "away_team": cand_row.get("away_team"),
                "ah_current": self._format_handicap_pair(
                    cand_row.get("ah_local"), cand_row.get("ah_away")
                ),
                "start_time": self._format_start_time(cand_row.get("start_time"))
                or (cand_row.get("time") if isinstance(cand_row.get("time"), str) else ""),
                "similarity_%": similarity,
            }
            result["handicap"] = (
                cand_row.get("handicap")
                if isinstance(cand_row.get("handicap"), str) and cand_row.get("handicap").strip()
                else result["ah_current"]
            )
            result["similarity_percent"] = similarity

            collected.append(result)
            update_payload = {"status": "update", **result}
            if callback:
                callback(update_payload)
            yield update_payload

        sorted_results = sorted(
            collected, key=lambda item: item["similarity_%"], reverse=True
        )

        if top_k is not None and top_k > 0:
            sorted_results = sorted_results[:top_k]

        self._last_results = sorted_results

        message = (
            "Busqueda completada. No se encontraron partidos similares."
            if not sorted_results
            else f"Busqueda completada. Coincidencias encontradas: {len(sorted_results)}"
        )
        summary = {
            "status": "complete",
            "seed_id": str(seed_id),
            "count": len(sorted_results),
            "results": sorted_results,
            "message": message,
        }
        if callback:
            callback(summary)
        yield summary

    def collect_top_matches(
        self, seed_id: str, top_k: Optional[int] = None
    ) -> List[Dict[str, object]]:
        """Ejecuta la busqueda y devuelve la lista final ya ordenada."""
        final_results: List[Dict[str, object]] = []
        for payload in self.incremental_search(seed_id, top_k=top_k):
            status = payload.get("status")
            if status == "complete":
                final_results = payload.get("results", [])
            elif status == "error":
                return []
        return final_results

    # --------------------------------------------------------------------- #
    # Utilidades varias
    # --------------------------------------------------------------------- #
    def _format_start_time(self, value: object) -> str:
        if isinstance(value, pd.Timestamp):
            if pd.isna(value):
                return ""
            return value.isoformat()
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, str):
            txt = value.strip()
            if not txt:
                return ""
            try:
                parsed = datetime.fromisoformat(txt)
                return parsed.isoformat()
            except ValueError:
                return txt
        return ""

    def _format_handicap_pair(self, local_value: object, away_value: object) -> str:
        local_fmt = (
            format_ah_as_decimal_string_of(str(local_value))
            if local_value is not None and not pd.isna(local_value)
            else "-"
        )
        away_fmt = (
            format_ah_as_decimal_string_of(str(away_value))
            if away_value is not None and not pd.isna(away_value)
            else "-"
        )
        return f"{local_fmt} / {away_fmt}" if away_fmt else local_fmt

    @property
    def last_results(self) -> List[Dict[str, object]]:
        return self._last_results


# ------------------------------------------------------------------------- #
# CLI helpers
# ------------------------------------------------------------------------- #

def _load_dataframes_from_json(data_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not data_path.exists():
        if not data_path.is_absolute():
            try:
                project_root = Path(__file__).resolve().parents[2]
                alt_path = project_root / data_path
            except IndexError:
                alt_path = data_path
            if alt_path.exists():
                data_path = alt_path
            else:
                raise FileNotFoundError(f"No se encontro el archivo de datos: {data_path}")
        else:
            raise FileNotFoundError(f"No se encontro el archivo de datos: {data_path}")

    with data_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    upcoming = pd.DataFrame.from_records(data.get("upcoming_matches", []))
    finished = pd.DataFrame.from_records(data.get("finished_matches", []))

    if upcoming.empty and finished.empty:
        raise ValueError("El archivo de datos no contiene partidos.")

    cache = pd.concat([finished, upcoming], ignore_index=True, sort=False)
    return cache, upcoming


def _print_cli_results(results: Iterable[Dict[str, object]]) -> None:
    results = list(results)
    if not results:
        print("No se encontraron partidos similares.")
        return

    for idx, match in enumerate(results, start=1):
        ah_current = match.get("ah_current", "-")
        start_time = match.get("start_time", "")
        similarity = match.get("similarity_%", 0.0)
        print(
            f"{idx:02d}. {match.get('home_team', '?')} vs {match.get('away_team', '?')} | "
            f"AH {ah_current} | {start_time} | {similarity:.2f}%"
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Filtro rapido de partidos similares por handicap asiatico."
    )
    parser.add_argument("--seed-id", required=True, help="ID del partido semilla.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Numero maximo de coincidencias a mostrar (0 = sin limite).",
    )
    parser.add_argument(
        "--data-file",
        default="data.json",
        help="Ruta al archivo JSON con 'upcoming_matches' y 'finished_matches'.",
    )

    args = parser.parse_args(argv)

    try:
        cache_df, upcoming_df = _load_dataframes_from_json(Path(args.data_file))
    except Exception as exc:  # pragma: no cover - errores de carga se reportan en CLI
        print(f"[ERROR] No fue posible cargar los datos: {exc}")
        return 1

    filter_tool = PatternFilter(cache_df, upcoming_df)
    matches = filter_tool.collect_top_matches(
        args.seed_id, top_k=None if args.top_k <= 0 else args.top_k
    )

    _print_cli_results(matches)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
