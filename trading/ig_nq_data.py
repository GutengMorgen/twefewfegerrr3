"""Minimal trading-ig helper for downloading NQ futures data."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from trading_ig import IGService
from trading_ig.rest import IGException


SHOW_PLOTS = False

DEFAULT_SEARCH_TERM = "NQ"
DEFAULT_EPIC = "IX.D.NASDAQ.IFMM.IP" #or IX.D.NASDAQ.FBMU1.IP have more spread
DEFAULT_RESOLUTION = "1Min"
DEFAULT_NUM_POINTS = 200
DEFAULT_EPICS = [
    "IX.D.NASDAQ.FBMU1.IP",
    "IX.D.NASDAQ.IFMM.IP",
]
DEFAULT_HISTORY_LOOKBACK_DAYS = 3
DEFAULT_HISTORY_CACHE_DIR = Path("data/history_cache")
DEFAULT_LOCAL_HISTORY_CSV = Path(__file__).resolve().parent / "data" / "US100_M1_202604200000_202604230936.csv"
DEFAULT_LOCAL_HISTORY_PARQUET = Path(__file__).resolve().parent / "data" / "US100_M1_202604200000_202604230936.parquet"
DEFAULT_CREDENTIALS_FILE = Path(__file__).resolve().with_name("ig_credentials.local.json")

ENV_IG_USERNAME = "IG_USERNAME"
ENV_IG_PASSWORD = "IG_PASSWORD"
ENV_IG_API_KEY = "IG_API_KEY"
ENV_IG_ACCOUNT_TYPE = "IG_ACCOUNT_TYPE"
ENV_IG_CREDENTIALS_FILE = "IG_CREDENTIALS_FILE"


@dataclass(frozen=True)
class Credentials:
    username: str
    password: str
    api_key: str
    account_type: str = "live"  # or "demo"


@dataclass(frozen=True)
class HistoricalOhlcvConfig:
    resolution: str = DEFAULT_RESOLUTION
    lookback_days: int = DEFAULT_HISTORY_LOOKBACK_DAYS
    cache_dir: Path = DEFAULT_HISTORY_CACHE_DIR
    source_csv_path: Path | None = DEFAULT_LOCAL_HISTORY_CSV


def _resolve_credentials_file(path_value: str | None) -> Path:
    if path_value is None or path_value.strip() == "":
        return DEFAULT_CREDENTIALS_FILE
    return Path(path_value).expanduser()


def _load_credentials_file(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid credentials JSON in {path}: {exc}") from exc
    except OSError as exc:
        raise OSError(f"Unable to read credentials file {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Credentials file {path} must contain a JSON object")
    return payload


def load_credentials(*, env: Mapping[str, str] | None = None) -> Credentials:
    env_values = os.environ if env is None else env

    username = str(env_values.get(ENV_IG_USERNAME, "")).strip()
    password = str(env_values.get(ENV_IG_PASSWORD, "")).strip()
    api_key = str(env_values.get(ENV_IG_API_KEY, "")).strip()
    account_type = str(env_values.get(ENV_IG_ACCOUNT_TYPE, "")).strip()

    credentials_file = _resolve_credentials_file(env_values.get(ENV_IG_CREDENTIALS_FILE))
    must_read_file = any(value == "" for value in (username, password, api_key, account_type))
    file_payload = _load_credentials_file(credentials_file) if must_read_file else {}

    if username == "":
        username = str(file_payload.get("username", "")).strip()
    if password == "":
        password = str(file_payload.get("password", "")).strip()
    if api_key == "":
        api_key = str(file_payload.get("api_key", "")).strip()
    if account_type == "":
        account_type = str(file_payload.get("account_type", "")).strip()

    missing_fields = [
        field_name
        for field_name, value in (
            ("username", username),
            ("password", password),
            ("api_key", api_key),
        )
        if value == ""
    ]
    if missing_fields:
        env_hint = ", ".join([ENV_IG_USERNAME, ENV_IG_PASSWORD, ENV_IG_API_KEY])
        raise RuntimeError(
            f"Missing IG credentials fields {missing_fields}. Set env vars ({env_hint}) "
            f"or provide file {credentials_file}."
        )

    normalized_account_type = account_type.lower() if account_type else "live"
    if normalized_account_type not in {"live", "demo"}:
        raise ValueError(
            f"{ENV_IG_ACCOUNT_TYPE} must be 'live' or 'demo'; received {account_type!r}"
        )

    return Credentials(
        username=username,
        password=password,
        api_key=api_key,
        account_type=normalized_account_type,
    )


def build_service(credentials: Credentials) -> IGService:
    service = IGService(
        credentials.username,
        credentials.password,
        credentials.api_key,
        acc_type=credentials.account_type,
        return_dataframe=True,
    )
    service.create_session()
    return service


def resolve_nq_future_epic(service: IGService, search_term: str = DEFAULT_SEARCH_TERM) -> tuple[str, dict[str, Any]]:
    markets = service.search_markets(search_term)
    if not isinstance(markets, pd.DataFrame):
        markets = pd.DataFrame(markets)

    if markets.empty:
        raise RuntimeError(f"No IG markets matched search term {search_term!r}.")

    frame = markets.copy()
    for column in ("epic", "instrumentType", "name", "instrumentName", "expiry"):
        if column not in frame.columns:
            frame[column] = ""

    instrument_type = frame["instrumentType"].astype(str).str.contains("FUT", case=False, na=False)
    search_text = frame[["name", "instrumentName", "epic"]].astype(str).agg(" ".join, axis=1)
    nq_text = search_text.str.contains(r"\bNQ\b|NASDAQ|NAS100", case=False, na=False)

    candidates = frame.loc[instrument_type & nq_text]
    if candidates.empty:
        candidates = frame.loc[instrument_type]
    if candidates.empty:
        candidates = frame

    selected = candidates.iloc[0].to_dict()
    epic = selected.get("epic")
    if not epic:
        raise RuntimeError(f"Could not resolve an epic from IG search results for {search_term!r}.")

    return str(epic), selected


def search_nq_markets(service: IGService, search_term: str = DEFAULT_SEARCH_TERM) -> pd.DataFrame:
    markets = service.search_markets(search_term)
    if not isinstance(markets, pd.DataFrame):
        markets = pd.DataFrame(markets)

    if markets.empty:
        raise RuntimeError(f"No IG markets matched search term {search_term!r}.")

    frame = markets.copy()
    for column in ("epic", "instrumentType", "name", "instrumentName", "expiry"):
        if column not in frame.columns:
            frame[column] = ""

    instrument_type = frame["instrumentType"].astype(str).str.contains("FUT", case=False, na=False)
    search_text = frame[["name", "instrumentName", "epic"]].astype(str).agg(" ".join, axis=1)
    nq_text = search_text.str.contains(r"\bNQ\b|NASDAQ|NAS100", case=False, na=False)

    matches = frame.loc[instrument_type & nq_text]
    if matches.empty:
        matches = frame.loc[instrument_type]
    if matches.empty:
        matches = frame

    return matches.reset_index(drop=True)


def get_market_by_epic(service: IGService, epic: str) -> dict[str, Any]:
    market = service.fetch_market_by_epic(epic)
    if isinstance(market, pd.DataFrame):
        if market.empty:
            raise RuntimeError(f"No IG market details returned for epic {epic!r}.")
        return market.iloc[0].to_dict()
    if isinstance(market, dict):
        return market
    return dict(market)


def fetch_raw_nq_history(
    service: IGService,
    epic: str,
    resolution: str = DEFAULT_RESOLUTION,
    num_points: int = DEFAULT_NUM_POINTS,
) -> pd.DataFrame:
    response = service.fetch_historical_prices_by_epic_and_num_points(
        epic,
        resolution,
        num_points,
    )
    prices = response["prices"]
    if not isinstance(prices, pd.DataFrame):
        prices = pd.DataFrame(prices)
    return prices


def _safe_history_slug(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def historical_ohlcv_window(
    *,
    lookback_days: int = DEFAULT_HISTORY_LOOKBACK_DAYS,
    now: pd.Timestamp | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    current_time = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if current_time.tzinfo is None:
        current_time = current_time.tz_localize("UTC")
    current_day_start = current_time.floor("D").tz_localize(None)
    start_time = current_day_start - pd.Timedelta(days=int(lookback_days))
    return start_time, current_day_start


def historical_ohlcv_cache_path(
    epic: str,
    resolution: str = DEFAULT_RESOLUTION,
    lookback_days: int = DEFAULT_HISTORY_LOOKBACK_DAYS,
    cache_dir: Path = DEFAULT_HISTORY_CACHE_DIR,
) -> Path:
    cache_name = f"{_safe_history_slug(epic)}_{_safe_history_slug(resolution)}_{int(lookback_days)}d.parquet"
    return cache_dir / cache_name


def normalize_ohlcv_history(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices.copy()
    if isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.reset_index()

    rename_map = {
        "DateTime": "time",
        "snapshotTimeUTC": "time",
        "snapshotTime": "time",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
    frame = frame.rename(columns=rename_map)

    if "time" not in frame.columns:
        if "index" in frame.columns:
            frame = frame.rename(columns={"index": "time"})
        else:
            raise ValueError("historical prices are missing a time column")

    if "volume" not in frame.columns:
        frame["volume"] = np.nan

    required_columns = ["time", "open", "high", "low", "close", "volume"]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"historical prices are missing required columns: {missing}")

    frame = frame.loc[:, required_columns].copy()
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame = frame.dropna(subset=["time"])
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values("time").drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)
    return frame


def load_ohlcv_from_csv(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path, sep="\t")
    rename_map = {
        "<DATE>": "date",
        "<TIME>": "time_of_day",
        "<OPEN>": "open",
        "<HIGH>": "high",
        "<LOW>": "low",
        "<CLOSE>": "close",
        "<TICKVOL>": "tick_volume",
        "<VOL>": "volume",
        "<SPREAD>": "spread",
    }
    frame = frame.rename(columns=rename_map)

    if "date" not in frame.columns or "time_of_day" not in frame.columns:
        raise ValueError(f"CSV history {csv_path} must include <DATE> and <TIME> columns")

    frame["time"] = pd.to_datetime(
        frame["date"].astype(str).str.strip() + " " + frame["time_of_day"].astype(str).str.strip(),
        format="%Y.%m.%d %H:%M:%S",
        errors="coerce",
    )
    if "volume" not in frame.columns or frame["volume"].isna().all():
        frame["volume"] = frame.get("tick_volume", pd.Series(dtype="float64"))

    ohlcv = frame[["time", "open", "high", "low", "close", "volume"]].copy()
    ohlcv["volume"] = pd.to_numeric(ohlcv["volume"], errors="coerce")
    return normalize_ohlcv_history(ohlcv)


def convert_ohlcv_csv_to_parquet(csv_path: Path, parquet_path: Path) -> pd.DataFrame:
    history = load_ohlcv_from_csv(csv_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    history.to_parquet(parquet_path, index=False)
    return history


def fetch_historical_ohlcv_by_epic(
    service: IGService,
    epic: str,
    resolution: str = DEFAULT_RESOLUTION,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    response = service.fetch_historical_prices_by_epic(
        epic,
        resolution=resolution,
        start_date=start_date.strftime("%Y-%m-%dT%H:%M:%S") if start_date is not None else None,
        end_date=end_date.strftime("%Y-%m-%dT%H:%M:%S") if end_date is not None else None,
        format=service.mid_prices,
    )
    prices = response["prices"] if isinstance(response, dict) else response
    if not isinstance(prices, pd.DataFrame):
        prices = pd.DataFrame(prices)
    return normalize_ohlcv_history(prices)


def load_cached_historical_ohlcv(cache_path: Path) -> pd.DataFrame | None:
    if not cache_path.exists():
        return None

    prices = pd.read_parquet(cache_path)
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        return None
    return normalize_ohlcv_history(prices)


def save_historical_ohlcv_cache(prices: pd.DataFrame, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(cache_path, index=False)


def load_or_fetch_historical_ohlcv(
    service: IGService,
    epic: str,
    config: HistoricalOhlcvConfig | None = None,
) -> pd.DataFrame:
    history_config = config or HistoricalOhlcvConfig()
    cache_path = historical_ohlcv_cache_path(
        epic,
        history_config.resolution,
        history_config.lookback_days,
        history_config.cache_dir,
    )
    cached = load_cached_historical_ohlcv(cache_path)
    if cached is not None and not cached.empty:
        latest_time = pd.Timestamp(cached["time"].max())
        if pd.notna(latest_time):
            freshness_cutoff = pd.Timestamp.now(tz="UTC").floor("D").tz_localize(None) - pd.Timedelta(days=1)
            if latest_time.normalize() >= freshness_cutoff:
                return cached

    if history_config.source_csv_path is not None and history_config.source_csv_path.exists():
        prices = convert_ohlcv_csv_to_parquet(history_config.source_csv_path, cache_path)
        return prices

    start_date, end_date = historical_ohlcv_window(lookback_days=history_config.lookback_days)
    prices = fetch_historical_ohlcv_by_epic(service, epic, history_config.resolution, start_date, end_date)
    save_historical_ohlcv_cache(prices, cache_path)
    return prices


def average_spread_for_epic(
    service: IGService,
    epic: str,
    resolution: str = DEFAULT_RESOLUTION,
    num_points: int = DEFAULT_NUM_POINTS,
) -> float:
    prices = fetch_raw_nq_history(service, epic, resolution, num_points)

    required_columns = [
        ("ask", "Open"),
        ("bid", "Open"),
        ("ask", "High"),
        ("bid", "High"),
        ("ask", "Low"),
        ("bid", "Low"),
        ("ask", "Close"),
        ("bid", "Close"),
    ]
    missing = [column for column in required_columns if column not in prices.columns]
    if missing:
        raise RuntimeError(f"Price data for {epic!r} is missing expected bid/ask columns: {missing}")

    spread = pd.concat(
        [
            prices[("ask", "Open")] - prices[("bid", "Open")],
            prices[("ask", "High")] - prices[("bid", "High")],
            prices[("ask", "Low")] - prices[("bid", "Low")],
            prices[("ask", "Close")] - prices[("bid", "Close")],
        ],
        axis=1,
    ).mean(axis=1)
    return float(spread.mean())


def average_liquidity_for_epic(
    service: IGService,
    epic: str,
    resolution: str = DEFAULT_RESOLUTION,
    num_points: int = DEFAULT_NUM_POINTS,
) -> float:
    prices = fetch_raw_nq_history(service, epic, resolution, num_points)
    volume_column = ("last", "Volume")
    if volume_column not in prices.columns:
        raise RuntimeError(f"Price data for {epic!r} is missing last/Volume.")
    volume = pd.to_numeric(prices[volume_column], errors="coerce")
    return float(volume.mean())


def instrument_name_for_epic(service: IGService, epic: str) -> str:
    market_details = get_market_by_epic(service, epic)
    instrument = market_details.get("instrument", {})
    if isinstance(instrument, dict):
        instrument_name = instrument.get("name", "")
    else:
        instrument_name = getattr(instrument, "name", "")
    if instrument_name:
        return str(instrument_name)
    for field in ("instrumentName", "name", "marketName"):
        value = market_details.get(field, "")
        if value:
            return str(value)
    return ""


def main() -> None:
    credentials = load_credentials()

    try:
        service = build_service(credentials)
    except IGException as exc:
        raise SystemExit(f"IG session failed: {exc}") from exc

    rows = []
    for epic in DEFAULT_EPICS:
        average_spread = average_spread_for_epic(service, epic, DEFAULT_RESOLUTION, DEFAULT_NUM_POINTS)
        average_liquidity = average_liquidity_for_epic(service, epic, DEFAULT_RESOLUTION, DEFAULT_NUM_POINTS)
        instrument_name = instrument_name_for_epic(service, epic)
        rows.append(
            {
                "epic": epic,
                "instrumentName": instrument_name,
                "average_spread": average_spread,
                "average_liquidity": average_liquidity,
            }
        )

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()