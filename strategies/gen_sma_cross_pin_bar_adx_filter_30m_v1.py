#!/usr/bin/env python3
"""Auto-generated: sma_cross trend + pin_bar entry + adx_filter regime on 30m"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "gen_sma_cross_pin_bar_adx_filter_30m_v1"
timeframe = "30m"
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

    sma_fast = close_s.rolling(10, min_periods=10).mean().values
    sma_slow = close_s.rolling(50, min_periods=50).mean().values
    trend = np.where(sma_fast > sma_slow, 1.0, -1.0)

    # ENTRY filter

    _body = np.abs(close - prices["open"].values) if "open" in prices.columns else np.ones(n)
    _range = high - low
    _upper_wick = high - np.maximum(close, prices["open"].values if "open" in prices.columns else close)
    _lower_wick = np.minimum(close, prices["open"].values if "open" in prices.columns else close) - low
    entry_ok_long = np.array([_lower_wick[i] > 2*_body[i] and _range[i]>0 for i in range(n)])
    entry_ok_short = np.array([_upper_wick[i] > 2*_body[i] and _range[i]>0 for i in range(n)])

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
