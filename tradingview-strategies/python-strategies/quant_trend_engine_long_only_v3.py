#!/usr/bin/env python3
"""
TradingView conversion: Quant Trend Engine Long Only v3
Compatibility: partial
Source Pine: bE05CfsO-BTC-USD-Quant-Trend-Engine-Long-Only-v3-BTC-USD-4H-Timeframe.pine

Adaptation:
- Orders are translated to next-bar target-position signals.
- Broker-managed stop orders and Friday close-on-bar behavior are approximated
  with bar-triggered exits that fill on the next bar open in this repo.
"""

import numpy as np
import pandas as pd

name = "tv_quant_trend_engine_long_only_v3"
timeframe = "4h"
leverage = 3.0

FAST_LEN = 18
MID_LEN = 50
SLOW_LEN = 120
SMOOTH_LEN = 3
PULLBACK_LEN = 8
BREAKOUT_LEN = 20
EFF_LEN = 18
PERSIST_LEN = 7
MOM_LEN = 12
SLOPE_LEN = 10
ATR_LEN = 14
ATR_BASE_LEN = 40

MIN_SCORE = 5.0
EXIT_SCORE = 2.5
MIN_SEP_PERC = 0.30
MIN_SLOW_SLOPE_PERC = 0.03
MIN_EFF = 0.33
MIN_ATR_REGIME = 0.95
MIN_BREAKOUT_ATR = 0.15
PULLBACK_ATR_MULT = 0.90
RECLAIM_ATR_MULT = 0.15
COOLDOWN_BARS = 5

HARD_STOP_PCT = 2.0
TRAIL_ATR_MULT = 2.8
PROFIT_LOCK_ATR_MULT = 20.8


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(values).ewm(span=span, adjust=False).mean().to_numpy(dtype=np.float64)


def _rma(values: np.ndarray, length: int) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float64)
    if len(values) < length:
        return out
    seed = np.nanmean(values[:length])
    out[length - 1] = seed
    alpha = 1.0 / length
    for i in range(length, len(values)):
        out[i] = out[i - 1] + alpha * (values[i] - out[i - 1])
    return out


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> np.ndarray:
    tr = np.zeros(len(close), dtype=np.float64)
    if len(close) == 0:
        return tr
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    return _rma(tr, length)


def _crossover(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(len(a), dtype=bool)
    if len(a) < 2:
        return out
    valid = np.isfinite(a) & np.isfinite(b)
    out[1:] = valid[1:] & valid[:-1] & (a[1:] > b[1:]) & (a[:-1] <= b[:-1])
    return out


def _crossunder(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(len(a), dtype=bool)
    if len(a) < 2:
        return out
    valid = np.isfinite(a) & np.isfinite(b)
    out[1:] = valid[1:] & valid[:-1] & (a[1:] < b[1:]) & (a[:-1] >= b[:-1])
    return out


def _bars_since(events: np.ndarray) -> np.ndarray:
    out = np.full(len(events), np.nan, dtype=np.float64)
    last = None
    for i, flag in enumerate(events):
        if flag:
            last = i
            out[i] = 0.0
        elif last is not None:
            out[i] = float(i - last)
    return out


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    open_ = prices["open"].to_numpy(dtype=np.float64)
    high = prices["high"].to_numpy(dtype=np.float64)
    low = prices["low"].to_numpy(dtype=np.float64)
    close = prices["close"].to_numpy(dtype=np.float64)
    n = len(close)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    fast = _ema(_ema(close, FAST_LEN), SMOOTH_LEN)
    mid = _ema(_ema(close, MID_LEN), SMOOTH_LEN)
    slow = _ema(_ema(close, SLOW_LEN), SMOOTH_LEN)

    bull_stack = (fast > mid) & (mid > slow)
    sep_perc = np.where(slow != 0.0, np.abs(fast - slow) / slow * 100.0, 0.0)
    sep_ok = sep_perc >= MIN_SEP_PERC

    fast_shift = np.roll(fast, SLOPE_LEN)
    mid_shift = np.roll(mid, SLOPE_LEN)
    slow_shift = np.roll(slow, SLOPE_LEN)
    fast_shift[:SLOPE_LEN] = np.nan
    mid_shift[:SLOPE_LEN] = np.nan
    slow_shift[:SLOPE_LEN] = np.nan
    fast_slope = np.where(np.isfinite(fast_shift) & (fast_shift != 0.0), (fast - fast_shift) / fast_shift * 100.0, np.nan)
    mid_slope = np.where(np.isfinite(mid_shift) & (mid_shift != 0.0), (mid - mid_shift) / mid_shift * 100.0, np.nan)
    slow_slope = np.where(np.isfinite(slow_shift) & (slow_shift != 0.0), (slow - slow_shift) / slow_shift * 100.0, np.nan)
    slope_ok = (slow_slope >= MIN_SLOW_SLOPE_PERC) & (mid_slope > 0.0) & (fast_slope > 0.0)

    close_eff_shift = np.roll(close, EFF_LEN)
    close_eff_shift[:EFF_LEN] = np.nan
    net_move = np.abs(close - close_eff_shift)
    diff_abs = np.abs(np.diff(close, prepend=np.nan))
    step_move = pd.Series(diff_abs).rolling(window=EFF_LEN, min_periods=EFF_LEN).sum().to_numpy(dtype=np.float64)
    efficiency = np.where(step_move > 0.0, net_move / step_move, 0.0)
    eff_ok = efficiency >= MIN_EFF

    close_diff_up = (pd.Series(close).diff() > 0).astype(float)
    up_bars = close_diff_up.rolling(window=PERSIST_LEN, min_periods=PERSIST_LEN).sum().to_numpy(dtype=np.float64)
    persist_ratio = np.where(np.isfinite(up_bars), up_bars / PERSIST_LEN, np.nan)
    close_mom_shift = np.roll(close, MOM_LEN)
    close_mom_shift[:MOM_LEN] = np.nan
    mom_raw = np.where(np.isfinite(close_mom_shift) & (close_mom_shift != 0.0), (close - close_mom_shift) / close_mom_shift * 100.0, np.nan)
    mom_ok = (mom_raw > 0.0) & (persist_ratio >= 0.57)

    atr = _atr(high, low, close, ATR_LEN)
    atr_base = pd.Series(atr).rolling(window=ATR_BASE_LEN, min_periods=ATR_BASE_LEN).mean().to_numpy(dtype=np.float64)
    atr_regime = np.where(np.isfinite(atr_base) & (atr_base != 0.0), atr / atr_base, 0.0)
    atr_ok = atr_regime >= MIN_ATR_REGIME

    hh = pd.Series(high).rolling(window=BREAKOUT_LEN, min_periods=BREAKOUT_LEN).max().shift(1).to_numpy(dtype=np.float64)
    breakout_dist = close - hh
    breakout_strength = np.where(np.isfinite(hh) & np.isfinite(atr) & (atr != 0.0), breakout_dist / atr, 0.0)
    breakout_ok = np.isfinite(hh) & (close > hh) & (breakout_strength >= MIN_BREAKOUT_ATR)

    pullback_low = pd.Series(low).rolling(window=PULLBACK_LEN, min_periods=PULLBACK_LEN).min().to_numpy(dtype=np.float64)
    dist_from_fast_atr = np.where(np.isfinite(atr) & (atr != 0.0), (fast - pullback_low) / atr, 0.0)
    deep_enough_pullback = dist_from_fast_atr >= PULLBACK_ATR_MULT
    close_prev = np.roll(close, 1)
    fast_prev = np.roll(fast, 1)
    mid_prev = np.roll(mid, 1)
    close_prev[0] = np.nan
    fast_prev[0] = np.nan
    mid_prev[0] = np.nan
    reclaim_fast = (close > fast) & (close_prev <= fast_prev)
    reclaim_mid = (close > mid) & (close_prev <= mid_prev)
    reclaim_strength = np.where(np.isfinite(atr) & (atr != 0.0), (close - fast) / atr, 0.0)
    reclaim_ok = (reclaim_fast | reclaim_mid) & (reclaim_strength >= RECLAIM_ATR_MULT)

    bull_cross = _crossover(fast, mid) | _crossover(fast, slow) | _crossover(mid, slow)
    bars_since_bull_cross = _bars_since(bull_cross)
    recent_trend_birth = np.isfinite(bars_since_bull_cross) & (bars_since_bull_cross <= 14.0)

    trend_score = (
        bull_stack.astype(float) * 1.50
        + sep_ok.astype(float) * 0.90
        + slope_ok.astype(float) * 1.10
        + eff_ok.astype(float) * 1.00
        + atr_ok.astype(float) * 0.80
        + mom_ok.astype(float) * 1.00
        + breakout_ok.astype(float) * 1.25
        + reclaim_ok.astype(float) * 1.10
    )

    trend_continuation_entry = bull_stack & breakout_ok & slope_ok & eff_ok & mom_ok
    pullback_reentry = bull_stack & sep_ok & slope_ok & deep_enough_pullback & reclaim_ok & eff_ok
    early_trend_entry = recent_trend_birth & bull_stack & sep_ok & slope_ok & atr_ok & mom_ok

    bear_cross = _crossunder(fast, mid) | _crossunder(fast, slow)
    structure_break = (close < mid) & (fast < mid)
    score_weak = trend_score <= EXIT_SCORE
    momentum_failure = (persist_ratio < 0.40) & (mom_raw < 0.0)
    regime_failure = (atr_regime < 0.80) & (efficiency < 0.25)

    open_time_ny = prices["open_time"].dt.tz_convert("America/New_York")
    ny_day = open_time_ny.dt.dayofweek.to_numpy()
    ny_hour = open_time_ny.dt.hour.to_numpy()
    ny_minute = open_time_ny.dt.minute.to_numpy()
    is_weekend = (ny_day == 5) | (ny_day == 6)
    is_friday_after_close = (ny_day == 4) & ((ny_hour > 16) | ((ny_hour == 16) & (ny_minute >= 0)))
    blocked_session = is_weekend | is_friday_after_close

    signals = np.zeros(n, dtype=np.float64)

    in_position = False
    pending_entry = False
    activate_at = -1
    entry_bar = -1
    entry_price = np.nan
    trail_stop = np.nan
    high_since_entry = np.nan
    last_exit_bar = None

    for i in range(n):
        if pending_entry and i == activate_at:
            in_position = True
            pending_entry = False
            entry_bar = i
            entry_price = open_[i]
            high_since_entry = high[i]
            trail_stop = np.nan

        target = 1.0 if (in_position or pending_entry) else 0.0

        if in_position:
            high_since_entry = max(high_since_entry, high[i]) if np.isfinite(high_since_entry) else high[i]
            if np.isfinite(atr[i]):
                raw_trail = close[i] - atr[i] * TRAIL_ATR_MULT
                profit_lock = high_since_entry - atr[i] * PROFIT_LOCK_ATR_MULT
                combined_trail = max(raw_trail, profit_lock)
                trail_stop = combined_trail if not np.isfinite(trail_stop) else max(trail_stop, combined_trail)

            hard_stop = entry_price * (1.0 - HARD_STOP_PCT / 100.0)
            stop_level = max(hard_stop, trail_stop) if np.isfinite(trail_stop) else hard_stop

            can_exit_now = i > entry_bar
            time_exit = can_exit_now and is_friday_after_close[i]
            stop_hit = can_exit_now and np.isfinite(stop_level) and (low[i] <= stop_level)
            exit_long = can_exit_now and (
                bear_cross[i]
                or structure_break[i]
                or score_weak[i]
                or momentum_failure[i]
                or regime_failure[i]
            )

            if time_exit or stop_hit or exit_long:
                target = 0.0
                in_position = False
                entry_bar = -1
                entry_price = np.nan
                trail_stop = np.nan
                high_since_entry = np.nan
                last_exit_bar = i

        if (not in_position) and (not pending_entry):
            cooldown_ok = last_exit_bar is None or (i - last_exit_bar > COOLDOWN_BARS)
            enter_long = (
                cooldown_ok
                and (not blocked_session[i])
                and np.isfinite(trend_score[i])
                and (trend_score[i] >= MIN_SCORE)
                and (close[i] > slow[i])
                and (
                    trend_continuation_entry[i]
                    or pullback_reentry[i]
                    or early_trend_entry[i]
                )
            )
            if enter_long:
                target = 1.0
                pending_entry = True
                activate_at = i + 1

        signals[i] = target

    return signals
