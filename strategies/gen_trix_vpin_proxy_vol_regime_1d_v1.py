#!/usr/bin/env python3
"""Auto-generated: trix trend + vpin_proxy entry + vol_regime regime on 1d"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "gen_trix_vpin_proxy_vol_regime_1d_v1"
timeframe = "1d"
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

    _e1 = close_s.ewm(span=12, min_periods=12, adjust=False).mean()
    _e2 = _e1.ewm(span=12, min_periods=12, adjust=False).mean()
    _e3 = _e2.ewm(span=12, min_periods=12, adjust=False).mean().values
    _trix = np.zeros(n)
    for i in range(1, n): _trix[i] = (_e3[i] - _e3[i-1]) / _e3[i-1] * 10000 if _e3[i-1] != 0 else 0
    _trix_signal = pd.Series(_trix).rolling(9, min_periods=9).mean().values
    trend = np.where(_trix > _trix_signal, 1.0, np.where(_trix < _trix_signal, -1.0, 0.0))

    # ENTRY filter

    # VPIN proxy via close location value — order flow imbalance
    _clv = np.where(high-low>0, (2*close-low-high)/(high-low), 0)
    _buy_vol = np.where(_clv > 0, volume * _clv, 0)
    _sell_vol = np.where(_clv < 0, volume * abs(_clv), 0)
    _buy_sum = pd.Series(_buy_vol).rolling(20, min_periods=20).sum().values
    _sell_sum = pd.Series(_sell_vol).rolling(20, min_periods=20).sum().values
    _total = _buy_sum + _sell_sum
    _imbalance = np.where(_total > 0, (_buy_sum - _sell_sum) / _total, 0)
    entry_ok_long = np.array([_imbalance[i] > 0.2 for i in range(n)])
    entry_ok_short = np.array([_imbalance[i] < -0.2 for i in range(n)])

    # REGIME filter

    _vol_tr = np.zeros(n)
    for i in range(1, n): _vol_tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    _vol_atr = pd.Series(_vol_tr).rolling(14, min_periods=14).mean().values
    _vol_pct = np.where(close > 0, _vol_atr / close, 0)
    _vol_pct_median = pd.Series(_vol_pct).rolling(100, min_periods=50).median().values
    regime_ok = np.array([not np.isnan(_vol_pct_median[i]) and _vol_pct[i] < _vol_pct_median[i] * 1.5 for i in range(n)])

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
