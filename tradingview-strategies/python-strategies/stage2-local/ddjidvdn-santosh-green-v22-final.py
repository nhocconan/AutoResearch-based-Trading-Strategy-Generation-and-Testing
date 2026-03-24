#!/usr/bin/env python3
"""
Yoona Green v22 (TradingView DdjIDVdn) converted to repo strategy format.

Classification: partial
- request.security HTF behavior is approximated via pandas resampling/ffill.
- strategy.exit intrabar order matching is approximated with OHLC checks.
- syminfo.pointvalue/mintick sizing is approximated from market price steps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

name = "santosh Green v22 - Final"
timeframe = "1m"
leverage = 1


def _wma(series: pd.Series, length: int) -> pd.Series:
    length = max(1, int(length))
    weights = np.arange(1, length + 1, dtype=float)
    return series.rolling(length, min_periods=length).apply(
        lambda x: float(np.dot(x, weights) / weights.sum()),
        raw=True,
    )


def _hma(series: pd.Series, length: int) -> pd.Series:
    length = max(1, int(length))
    half = max(1, int(round(length / 2.0)))
    root = max(1, int(round(np.sqrt(length))))
    return _wma(2.0 * _wma(series, half) - _wma(series, length), root)


def _rma(series: pd.Series, length: int) -> pd.Series:
    length = max(1, int(length))
    return series.ewm(alpha=1.0 / length, adjust=False).mean()


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff().fillna(0.0)
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    ru = _rma(up, length)
    rd = _rma(down, length)
    rs = ru / rd.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    fallback = pd.Series(np.where(rd == 0.0, 100.0, 0.0), index=close.index, dtype=float)
    out = out.where(out.notna(), fallback)
    return out.astype(float)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return _rma(tr.fillna(0.0), length)


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - macd_signal
    return macd_line, macd_signal, hist


def _vwap_ny_session(prices: pd.DataFrame, typical: pd.Series) -> pd.Series:
    ts = pd.to_datetime(prices["open_time"], utc=True)
    ny_day = ts.dt.tz_convert("America/New_York").dt.date
    num = (typical * prices["volume"]).groupby(ny_day).cumsum()
    den = prices["volume"].groupby(ny_day).cumsum().replace(0.0, np.nan)
    return (num / den).ffill().bfill()


def _timeframe_to_rule(tf: str) -> str:
    raw = (tf or "").strip().lower()
    if raw.endswith("m"):
        return raw
    if raw.endswith("h"):
        mins = int(raw[:-1]) * 60
        return f"{mins}min"
    if raw.isdigit():
        return f"{int(raw)}min"
    return "30min"


def _security_close(prices: pd.DataFrame, tf: str) -> pd.Series:
    s = pd.Series(prices["close"].to_numpy(dtype=float), index=pd.to_datetime(prices["open_time"], utc=True))
    rule = _timeframe_to_rule(tf)
    htf = s.resample(rule, label="right", closed="right").last()
    aligned = htf.reindex(s.index, method="ffill")
    return aligned.fillna(s).reset_index(drop=True)


def _inferred_tick(close: np.ndarray) -> float:
    if close.size < 2:
        return 0.01
    diffs = np.abs(np.diff(close))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        return 0.01
    return float(np.percentile(diffs, 10))


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    n = len(prices)
    if n == 0:
        return np.array([], dtype=float)

    close = prices["close"].astype(float).reset_index(drop=True)
    high = prices["high"].astype(float).reset_index(drop=True)
    low = prices["low"].astype(float).reset_index(drop=True)
    volume = prices["volume"].astype(float).reset_index(drop=True)

    # Pine defaults.
    i_run_all_day = False
    i_ny_on, i_ny_from, i_ny_to = False, 830, 1100
    i_au_on, i_au_from, i_au_to = False, 1800, 1900
    i_as_on, i_as_from, i_as_to = False, 2000, 2100
    i_lo_on, i_lo_from, i_lo_to = True, 200, 300
    i_cu_on, i_cu_from, i_cu_to = False, 1500, 1545

    i_exit_445 = True
    i_htf1_on, i_htf1_tf, i_htf1_len = True, "30", 25
    i_htf2_on, i_htf2_tf, i_htf2_len = False, "60", 25
    i_htf3_on, i_htf3_tf, i_htf3_len = False, "240", 20

    i_vwap_on, i_vwap_mult = False, 1.0
    i_macd_on, i_pa_on, i_vol_on = False, False, False
    i_ema_on, i_ema_fast, i_ema_slow = True, 9, 13
    i_rsi_on, i_rsi_long, i_rsi_short = True, 60.0, 40.0
    i_tp_usd, i_sl_usd = 30.0, 150.0
    i_atr_len, i_atr_mult = 10, 3.5

    # Point/tick approximation for non-TradingView execution.
    pt_val = 1.0
    tick_val = max(_inferred_tick(close.to_numpy()), 1e-8)
    tp_ticks = int(round(i_tp_usd / pt_val / tick_val))
    sl_ticks = int(round(i_sl_usd / pt_val / tick_val))
    tp_dist = max(tp_ticks, 1) * tick_val
    sl_dist = max(sl_ticks, 1) * tick_val

    # HTF via resampling + ffill (lookahead_off approximation).
    htf1_close = _security_close(prices, i_htf1_tf)
    htf2_close = _security_close(prices, i_htf2_tf)
    htf3_close = _security_close(prices, i_htf3_tf)
    hma1 = _hma(htf1_close, i_htf1_len)
    hma2 = _hma(htf2_close, i_htf2_len)
    hma3 = _hma(htf3_close, i_htf3_len)

    hma1_up = (hma1 > hma1.shift(1)).fillna(False)
    hma2_up = (hma2 > hma2.shift(1)).fillna(False)
    hma3_up = (hma3 > hma3.shift(1)).fillna(False)
    hma1_dn = (hma1 < hma1.shift(1)).fillna(False)
    hma2_dn = (hma2 < hma2.shift(1)).fillna(False)
    hma3_dn = (hma3 < hma3.shift(1)).fillna(False)
    htf_bull = ((hma1_up if i_htf1_on else True) & (hma2_up if i_htf2_on else True) & (hma3_up if i_htf3_on else True)).fillna(False)
    htf_bear = ((hma1_dn if i_htf1_on else True) & (hma2_dn if i_htf2_on else True) & (hma3_dn if i_htf3_on else True)).fillna(False)

    ema_f = close.ewm(span=i_ema_fast, adjust=False).mean()
    ema_s = close.ewm(span=i_ema_slow, adjust=False).mean()
    ema_bull = (ema_f > ema_s) if i_ema_on else pd.Series(True, index=close.index)
    ema_bear = (ema_f < ema_s) if i_ema_on else pd.Series(True, index=close.index)
    ema_cross_up = ((ema_f > ema_s) & (ema_f.shift(1) <= ema_s.shift(1))).fillna(False)
    ema_cross_dn = ((ema_f < ema_s) & (ema_f.shift(1) >= ema_s.shift(1))).fillna(False)

    rsi = _rsi(close, 14)
    rsi_ok_long = (rsi < i_rsi_long) if i_rsi_on else pd.Series(True, index=close.index)
    rsi_ok_shrt = (rsi > i_rsi_short) if i_rsi_on else pd.Series(True, index=close.index)

    vwap = _vwap_ny_session(prices, (high + low + close) / 3.0)
    vwap_bull = (close > vwap * i_vwap_mult) if i_vwap_on else pd.Series(True, index=close.index)
    vwap_bear = (close < vwap * i_vwap_mult) if i_vwap_on else pd.Series(True, index=close.index)

    macd_line, macd_sig, _ = _macd(close)
    macd_bull = (macd_line > macd_sig) if i_macd_on else pd.Series(True, index=close.index)
    macd_bear = (macd_line < macd_sig) if i_macd_on else pd.Series(True, index=close.index)

    vol_avg = volume.rolling(20, min_periods=1).mean()
    vol_ok = (volume > vol_avg) if i_vol_on else pd.Series(True, index=close.index)
    pa_bull = ((high > high.shift(1)) & (low > low.shift(1))) if i_pa_on else pd.Series(True, index=close.index)
    pa_bear = ((high < high.shift(1)) & (low < low.shift(1))) if i_pa_on else pd.Series(True, index=close.index)

    atr = _atr(high, low, close, i_atr_len)
    atr_trail = atr * i_atr_mult

    ts = pd.to_datetime(prices["open_time"], utc=True).dt.tz_convert("America/New_York")
    hhmm = ts.dt.hour * 100 + ts.dt.minute
    is_445pm = (ts.dt.hour == 16) & (ts.dt.minute == 45)

    def in_win(start_hhmm: int, end_hhmm: int) -> pd.Series:
        return (hhmm >= start_hhmm) & (hhmm <= end_hhmm)

    in_ny = i_ny_on and in_win(i_ny_from, i_ny_to)
    in_au = i_au_on and in_win(i_au_from, i_au_to)
    in_as = i_as_on and in_win(i_as_from, i_as_to)
    in_lo = i_lo_on and in_win(i_lo_from, i_lo_to)
    in_cu = i_cu_on and in_win(i_cu_from, i_cu_to)
    in_session = pd.Series(bool(i_run_all_day), index=close.index) | in_ny | in_au | in_as | in_lo | in_cu

    long_sig = (ema_cross_up & htf_bull & ema_bull & rsi_ok_long & vwap_bull & macd_bull & vol_ok & pa_bull).fillna(False)
    shrt_sig = (ema_cross_dn & htf_bear & ema_bear & rsi_ok_shrt & vwap_bear & macd_bear & vol_ok & pa_bear).fillna(False)

    signals = np.zeros(n, dtype=float)
    pos = 0
    entry_price = np.nan

    for i in range(n):
        c = float(close.iloc[i])
        h = float(high.iloc[i])
        l = float(low.iloc[i])

        if pos > 0:
            fixed_stop = float(entry_price - sl_dist)
            trail_stop = float(c - atr_trail.iloc[i]) if np.isfinite(atr_trail.iloc[i]) else fixed_stop
            stop_level = max(fixed_stop, trail_stop)
            tp_level = float(entry_price + tp_dist)
            if l <= stop_level or h >= tp_level:
                pos = 0
                entry_price = np.nan
        elif pos < 0:
            fixed_stop = float(entry_price + sl_dist)
            trail_stop = float(c + atr_trail.iloc[i]) if np.isfinite(atr_trail.iloc[i]) else fixed_stop
            stop_level = min(fixed_stop, trail_stop)
            tp_level = float(entry_price - tp_dist)
            if h >= stop_level or l <= tp_level:
                pos = 0
                entry_price = np.nan

        if i_exit_445 and bool(is_445pm.iloc[i]) and pos != 0:
            pos = 0
            entry_price = np.nan

        can_trade = bool(in_session.iloc[i]) and (pos == 0) and (not bool(is_445pm.iloc[i]))
        if can_trade and bool(long_sig.iloc[i]):
            pos = 1
            entry_price = c
        elif can_trade and bool(shrt_sig.iloc[i]):
            pos = -1
            entry_price = c

        signals[i] = float(pos)

    return signals
