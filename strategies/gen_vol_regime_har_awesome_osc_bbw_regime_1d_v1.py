#!/usr/bin/env python3
"""Auto-generated: vol_regime_har trend + awesome_osc entry + bbw_regime regime on 1d"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "gen_vol_regime_har_awesome_osc_bbw_regime_1d_v1"
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

    # HAR volatility regime: realized vol predicts strategy mode
    _ret2 = pd.Series(close).pct_change().values ** 2
    _rv1d = pd.Series(_ret2).rolling(6, min_periods=6).sum().values
    _rv5d = pd.Series(_ret2).rolling(30, min_periods=30).sum().values
    _rv22d = pd.Series(_ret2).rolling(132, min_periods=132).sum().values
    _rv_pred = np.zeros(n)
    for i in range(132+1, n):
        _rv_pred[i] = 0.4*_rv1d[i-1] + 0.3*_rv5d[i-1] + 0.3*_rv22d[i-1]
    _rv_median = pd.Series(_rv_pred).rolling(100, min_periods=50).median().values
    # Low vol predicted = mean revert (long dips), high vol = trend follow
    _ema_f = close_s.ewm(span=21, min_periods=21, adjust=False).mean().values
    _ema_s = close_s.ewm(span=55, min_periods=55, adjust=False).mean().values
    trend = np.zeros(n)
    for i in range(100, n):
        if np.isnan(_rv_median[i]) or _rv_median[i]==0: continue
        if _rv_pred[i] < _rv_median[i]:  # low vol regime
            trend[i] = 1.0 if close[i] < _ema_f[i] else (-1.0 if close[i] > _ema_s[i] else 0.0)
        else:  # high vol regime
            trend[i] = 1.0 if _ema_f[i] > _ema_s[i] else (-1.0 if _ema_f[i] < _ema_s[i] else 0.0)

    # ENTRY filter

    _ao_fast = pd.Series((high+low)/2).rolling(5,min_periods=5).mean().values
    _ao_slow = pd.Series((high+low)/2).rolling(34,min_periods=34).mean().values
    _ao = _ao_fast - _ao_slow
    entry_ok_long = np.array([_ao[i]>0 and (i<1 or _ao[i]>_ao[i-1]) for i in range(n)])
    entry_ok_short = np.array([_ao[i]<0 and (i<1 or _ao[i]<_ao[i-1]) for i in range(n)])

    # REGIME filter

    _sma = close_s.rolling(20, min_periods=20).mean().values
    _std = close_s.rolling(20, min_periods=20).std().values
    _bbw = np.where(_sma > 0, _std / _sma, 0)
    _bbw_pct = pd.Series(_bbw).rolling(100, min_periods=50).rank(pct=True).values
    regime_ok = np.array([not np.isnan(_bbw_pct[i]) and _bbw_pct[i] < 0.7 and _bbw_pct[i] > 0.1 for i in range(n)])

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
