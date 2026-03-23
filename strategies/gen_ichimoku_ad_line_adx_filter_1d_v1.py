#!/usr/bin/env python3
"""Auto-generated: ichimoku trend + ad_line entry + adx_filter regime on 1d"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "gen_ichimoku_ad_line_adx_filter_1d_v1"
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

    tenkan = (pd.Series(high).rolling(20).max().values + pd.Series(low).rolling(20).min().values) / 2
    kijun = (pd.Series(high).rolling(60).max().values + pd.Series(low).rolling(60).min().values) / 2
    trend = np.zeros(n)
    for i in range(60, n):
        if not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
            trend[i] = 1.0 if tenkan[i] > kijun[i] and close[i] > kijun[i] else (-1.0 if tenkan[i] < kijun[i] and close[i] < kijun[i] else 0.0)

    # ENTRY filter

    _clv = np.where(high-low>0, (2*close-low-high)/(high-low), 0)
    _ad = np.cumsum(_clv * volume)
    _ad_ema = pd.Series(_ad).ewm(span=21,min_periods=21,adjust=False).mean().values
    entry_ok_long = np.array([_ad[i]>_ad_ema[i] if not np.isnan(_ad_ema[i]) else False for i in range(n)])
    entry_ok_short = np.array([_ad[i]<_ad_ema[i] if not np.isnan(_ad_ema[i]) else False for i in range(n)])

    # REGIME filter

    _pdm = np.zeros(n); _ndm = np.zeros(n)
    for i in range(1, n):
        hd = high[i]-high[i-1]; ld = low[i-1]-low[i]
        if hd > ld and hd > 0: _pdm[i] = hd
        if ld > hd and ld > 0: _ndm[i] = ld
    _tr2 = np.zeros(n)
    for i in range(1, n): _tr2[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    _atr2 = pd.Series(_tr2).ewm(span=14, min_periods=14, adjust=False).mean().values
    _pdi = np.where(_atr2>0, 100*pd.Series(_pdm).ewm(span=14,min_periods=14,adjust=False).mean().values/_atr2, 0)
    _ndi = np.where(_atr2>0, 100*pd.Series(_ndm).ewm(span=14,min_periods=14,adjust=False).mean().values/_atr2, 0)
    _dx = np.where(_pdi+_ndi>0, 100*np.abs(_pdi-_ndi)/(_pdi+_ndi), 0)
    adx = pd.Series(_dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    regime_ok = np.array([adx[i] > 20 for i in range(n)])

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
