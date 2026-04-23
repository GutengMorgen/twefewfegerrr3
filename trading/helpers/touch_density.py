from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


LEVEL_MULTIPLIERS = np.asarray((0.5, 1.0, 1.5), dtype=np.float64)


@dataclass(frozen=True)
class TouchDensitySeries:
    signal: np.ndarray
    long_density: np.ndarray
    short_density: np.ndarray


@dataclass(frozen=True)
class DailyStatsSeries:
    day_id: np.ndarray
    day_start_indices: np.ndarray
    day_open: np.ndarray
    day_close: np.ndarray
    daily_change: np.ndarray
    rolling_mean: np.ndarray
    rolling_std: np.ndarray


def _day_id_array(time_values: np.ndarray) -> np.ndarray:
    return time_values.astype("datetime64[D]").astype(np.int64)


def _minute_of_day_array(time_values: np.ndarray) -> np.ndarray:
    minutes = time_values.astype("datetime64[m]").astype(np.int64)
    return (minutes % (24 * 60)).astype(np.int32)


def _day_boundaries(day_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if day_id.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty

    change_points = np.flatnonzero(day_id[1:] != day_id[:-1]) + 1
    starts = np.concatenate((np.array([0], dtype=np.int64), change_points.astype(np.int64)))
    ends = np.concatenate((change_points.astype(np.int64), np.array([day_id.size], dtype=np.int64)))
    return starts, ends


def _rolling_sample_mean_std(values: np.ndarray, lookback_days: int) -> tuple[np.ndarray, np.ndarray]:
    if lookback_days < 1:
        raise ValueError("lookback_days must be >= 1")

    count = int(values.size)
    rolling_mean = np.full(count, np.nan, dtype=np.float64)
    rolling_std = np.full(count, np.nan, dtype=np.float64)

    if count == 0:
        return rolling_mean, rolling_std

    csum = np.cumsum(values, dtype=np.float64)
    csum_sq = np.cumsum(values * values, dtype=np.float64)

    for index in range(count):
        start = max(0, index - lookback_days + 1)
        window_count = index - start + 1
        window_total = csum[index] - (csum[start - 1] if start > 0 else 0.0)
        window_total_sq = csum_sq[index] - (csum_sq[start - 1] if start > 0 else 0.0)

        mean = window_total / float(window_count)
        rolling_mean[index] = mean

        if window_count < 2:
            continue

        population_var = (window_total_sq / float(window_count)) - (mean * mean)
        if population_var < 0.0:
            population_var = 0.0
        rolling_std[index] = float(np.sqrt(population_var * float(window_count) / float(window_count - 1)))

    return rolling_mean, rolling_std


def compute_daily_change_stats(
    open_values: np.ndarray,
    close_values: np.ndarray,
    day_id: np.ndarray,
    lookback_days: int,
) -> DailyStatsSeries:
    if open_values.ndim != 1 or close_values.ndim != 1 or day_id.ndim != 1:
        raise ValueError("open_values, close_values, and day_id must be 1D arrays")
    if open_values.size != close_values.size or open_values.size != day_id.size:
        raise ValueError("open_values, close_values, and day_id must have the same length")

    day_starts, day_ends = _day_boundaries(day_id)
    if day_starts.size == 0:
        empty_float = np.empty(0, dtype=np.float64)
        empty_int = np.empty(0, dtype=np.int64)
        return DailyStatsSeries(empty_int, empty_int, empty_float, empty_float, empty_float, empty_float, empty_float)

    day_open = np.asarray(open_values[day_starts], dtype=np.float64)
    day_close = np.asarray(close_values[day_ends - 1], dtype=np.float64)
    safe_open = np.where(day_open != 0.0, day_open, np.nan)
    daily_change = np.where(np.isfinite(safe_open), (day_close - day_open) / safe_open, 0.0)
    rolling_mean, rolling_std = _rolling_sample_mean_std(daily_change, lookback_days)

    return DailyStatsSeries(
        day_id=np.asarray(day_id[day_starts], dtype=np.int64),
        day_start_indices=np.asarray(day_starts, dtype=np.int64),
        day_open=day_open,
        day_close=day_close,
        daily_change=daily_change,
        rolling_mean=rolling_mean,
        rolling_std=rolling_std,
    )


def build_std_levels(
    base_price: float,
    mean_value: float,
    std_value: float,
    multipliers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if multipliers.ndim != 1:
        raise ValueError("multipliers must be a 1D array")

    upper_factor = (std_value + mean_value) * base_price
    lower_factor = (std_value - mean_value) * base_price

    upper_levels = base_price + (upper_factor * multipliers)
    lower_levels = base_price - (lower_factor * multipliers)
    return upper_levels.astype(np.float64, copy=False), lower_levels.astype(np.float64, copy=False)


def _gaussian_kernel(width: int) -> np.ndarray:
    if width < 1:
        raise ValueError("kernel_width must be >= 1")

    radius = max(1, int(width))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    sigma = max(1.0, radius / 2.0)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    return kernel / float(np.sum(kernel))


def _center_crop(values: np.ndarray, target_size: int) -> np.ndarray:
    if int(values.size) == int(target_size):
        return values
    if int(values.size) < int(target_size):
        raise ValueError("cannot crop an array shorter than the target size")

    start = (int(values.size) - int(target_size)) // 2
    end = start + int(target_size)
    return values[start:end]


def compute_touch_density_from_bars(
    bars: pd.DataFrame,
    *,
    lookback_days: int,
    kernel_width: int,
    signal_timeframe_minutes: int,
) -> TouchDensitySeries:
    if bars.empty:
        empty = np.zeros(0, dtype=np.float64)
        return TouchDensitySeries(signal=empty, long_density=empty, short_density=empty)

    required_columns = {"time", "open", "high", "low", "close"}
    missing = required_columns.difference(bars.columns)
    if missing:
        raise ValueError(f"bars is missing required columns: {sorted(missing)}")

    ordered = bars.sort_values("time").reset_index(drop=True)
    time_values = pd.to_datetime(ordered["time"], errors="coerce").to_numpy(dtype="datetime64[us]")
    open_values = np.asarray(ordered["open"], dtype=np.float64)
    high_values = np.asarray(ordered["high"], dtype=np.float64)
    low_values = np.asarray(ordered["low"], dtype=np.float64)
    close_values = np.asarray(ordered["close"], dtype=np.float64)

    if time_values.size == 0:
        empty = np.zeros(0, dtype=np.float64)
        return TouchDensitySeries(signal=empty, long_density=empty, short_density=empty)

    minute_of_day = _minute_of_day_array(time_values)
    day_id = _day_id_array(time_values)

    if signal_timeframe_minutes <= 1:
        htf_time_values = time_values
        htf_open_values = open_values
        htf_high_values = high_values
        htf_low_values = low_values
        htf_close_values = close_values
        htf_minute_of_day = minute_of_day
        htf_day_id = day_id
    else:
        bucket_index = minute_of_day // int(signal_timeframe_minutes)
        bucket_change = (day_id[1:] != day_id[:-1]) | (bucket_index[1:] != bucket_index[:-1])
        boundaries = np.flatnonzero(bucket_change) + 1
        starts = np.concatenate((np.array([0], dtype=np.int64), boundaries.astype(np.int64)))
        ends = np.concatenate((boundaries.astype(np.int64), np.array([time_values.size], dtype=np.int64)))

        count = int(starts.size)
        htf_time_values = np.empty(count, dtype=time_values.dtype)
        htf_open_values = np.empty(count, dtype=np.float64)
        htf_high_values = np.empty(count, dtype=np.float64)
        htf_low_values = np.empty(count, dtype=np.float64)
        htf_close_values = np.empty(count, dtype=np.float64)
        htf_minute_of_day = np.empty(count, dtype=np.int32)
        htf_day_id = np.empty(count, dtype=np.int64)

        for i in range(count):
            start = int(starts[i])
            end = int(ends[i])
            htf_time_values[i] = time_values[end - 1]
            htf_open_values[i] = float(open_values[start])
            htf_high_values[i] = float(np.max(high_values[start:end]))
            htf_low_values[i] = float(np.min(low_values[start:end]))
            htf_close_values[i] = float(close_values[end - 1])
            htf_minute_of_day[i] = int(minute_of_day[end - 1])
            htf_day_id[i] = int(day_id[end - 1])

    daily_stats = compute_daily_change_stats(
        open_values=htf_open_values,
        close_values=htf_close_values,
        day_id=htf_day_id,
        lookback_days=lookback_days,
    )

    signal = np.zeros(htf_close_values.size, dtype=np.float64)
    long_density = np.zeros(htf_close_values.size, dtype=np.float64)
    short_density = np.zeros(htf_close_values.size, dtype=np.float64)

    if htf_close_values.size == 0 or daily_stats.day_open.size < 2:
        return TouchDensitySeries(signal=signal, long_density=long_density, short_density=short_density)

    kernel = _gaussian_kernel(kernel_width)
    day_starts, _ = _day_boundaries(htf_day_id)

    for day_index in range(1, int(daily_stats.day_open.size)):
        base_price = float(daily_stats.day_open[day_index])
        mean_value = float(daily_stats.rolling_mean[day_index - 1])
        std_value = float(daily_stats.rolling_std[day_index - 1])
        if not np.isfinite(mean_value) or not np.isfinite(std_value) or std_value <= 0.0:
            continue

        day_start = int(day_starts[day_index])
        day_end = int(day_starts[day_index + 1]) if day_index + 1 < day_starts.size else int(htf_close_values.size)
        if day_end <= day_start:
            continue

        day_high = htf_high_values[day_start:day_end]
        day_low = htf_low_values[day_start:day_end]

        upper_levels, lower_levels = build_std_levels(base_price, mean_value, std_value, LEVEL_MULTIPLIERS)
        long_binary = (day_high >= float(upper_levels[1])).astype(np.float64)
        short_binary = (day_low <= float(lower_levels[1])).astype(np.float64)

        long_smooth = _center_crop(np.convolve(long_binary, kernel, mode="same"), day_end - day_start)
        short_smooth = _center_crop(np.convolve(short_binary, kernel, mode="same"), day_end - day_start)

        long_density[day_start:day_end] = long_smooth
        short_density[day_start:day_end] = short_smooth
        signal[day_start:day_end] = long_smooth - short_smooth

    return TouchDensitySeries(signal=signal, long_density=long_density, short_density=short_density)