#!/usr/bin/env python3
"""
Experiment #028: 6h Donchian Breakout + Volume + 1d EMA21 Trend + Choppiness Filter

HYPOTHESIS: Simple price channel breakout with volume confirmation is the most
robust trading signal. By using 1d EMA21 for trend direction and Choppiness
Index to filter range-bound markets, this strategy captures institutional
breakouts while avoiding whipsaws in both bull and bear markets.

KEY IMPROVEMENTS OVER FAILED #027 (0 trades):
- Removed choppiness from entry conditions (too restrictive → 0 trades)
- Keep choppiness only as ENTRY FILTER (skip in very choppy markets)
- Loosened volume requirement to 1.3x for more signals
- Using EMA21 instead of SMA50 for faster trend response
- 50-bar Donchian for clearer, more significant breakouts

TARGET: 60-120 total trades over 4 years = 15-30/year. HARD MAX: 200.
Signal size: 0.30 (conservative).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_vol_ema21_chop_v2"
timeframe = "6h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - lower = trending, higher = choppy"""
    n = len(high)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            if j > 0:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]))
            else:
                tr = high[j] - low[j]
            tr_sum += tr
        
        if tr_sum > 0:
            hh = np.max(high[i - period + 1:i + 1])
            ll = np.min(low[i - period + 1:i + 1])
            range_hl = hh - ll
            
            if range_hl > 0:
                chop[i] = 100 * np.log10(tr_sum / range_hl) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values