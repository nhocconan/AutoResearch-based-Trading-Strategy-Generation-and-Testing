#!/usr/bin/env python3
"""Auto-generated: heikin_ashi trend + rvi entry + keltner_squeeze regime on 1h"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "gen_heikin_ashi_rvi_keltner_squeeze_1h_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    close_s = pd.Series(close)

    # ATR for stoploss
    _tr = np.zeros(n)
    for i in range(1, n): _tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    atr = pd.Series(_tr).rolling(14, min_periods=14).mean().values

    # TREND indicator

    ha_close = (prices["open"].values + high + low + close) / 4
    ha_open = np.zeros(n); ha_open[0] = (prices["open"].values[0] + close[0]) / 2
    for i in range(1, n): ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
    ha_ema = pd.Series(ha_close).ewm(span=10, min_periods=10, adjust=False).mean().values
    trend = np.where(ha_close > ha_ema, 1.0, np.where(ha_close < ha_ema, -1.0, 0.0))

    # ENTRY filter

    _rvi_num = pd.Series((close - prices["open"].values) if "open" in prices.columns else np.zeros(n)).rolling(10,min_periods=10).mean().values
    _rvi_den = pd.Series(high - low).rolling(10,min_periods=10).mean().values
    _rvi = np.where(_rvi_den>0, _rvi_num/_rvi_den, 0)
    _rvi_sig = pd.Series(_rvi).rolling(4,min_periods=4).mean().values
    entry_ok_long = np.array([_rvi[i]>_rvi_sig[i] for i in range(n)])
    entry_ok_short = np.array([_rvi[i]<_rvi_sig[i] for i in range(n)])

    # REGIME filter

    _kc_mid = close_s.ewm(span=20, min_periods=20, adjust=False).mean().values
    _kc_tr = np.zeros(n)
    for i in range(1, n): _kc_tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    _kc_atr = pd.Series(_kc_tr).rolling(20, min_periods=20).mean().values
    _kc_upper = _kc_mid + 1.5 * _kc_atr; _kc_lower = _kc_mid - 1.5 * _kc_atr
    _bb_mid = close_s.rolling(20, min_periods=20).mean().values
    _bb_std = close_s.rolling(20, min_periods=20).std().values
    _bb_upper = _bb_mid + 2 * _bb_std; _bb_lower = _bb_mid - 2 * _bb_std
    regime_ok = np.array([not np.isnan(_kc_upper[i]) and _bb_upper[i] < _kc_upper[i] for i in range(n)])

    signals = np.zeros(n)
    SIZE = 0.25
    entry_price = 0.0
    in_trade = 0

    for i in range(100, n):
        if np.isnan(atr[i]) or atr[i] == 0: continue

        # Manage position
        if in_trade != 0:
            if in_trade == 1 and close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0; in_trade = 0; continue
            if in_trade == -1 and close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0; in_trade = 0; continue
            if in_trade == 1 and trend[i] < 0:
                signals[i] = 0.0; in_trade = 0; continue
            if in_trade == -1 and trend[i] > 0:
                signals[i] = 0.0; in_trade = 0; continue
            signals[i] = SIZE * in_trade; continue

        if not regime_ok[i]: signals[i] = 0.0; continue

        if trend[i] > 0 and entry_ok_long[i]:
            signals[i] = SIZE; entry_price = close[i]; in_trade = 1
        elif trend[i] < 0 and entry_ok_short[i]:
            signals[i] = -SIZE; entry_price = close[i]; in_trade = -1
        else:
            signals[i] = 0.0

    return signals
