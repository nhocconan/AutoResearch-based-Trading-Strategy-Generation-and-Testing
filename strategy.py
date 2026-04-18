# %pip install git+https://github.com/upbit-oss/ta-momentum.git
#!/usr/bin/env python3
"""
12h_TRIX_Volume_Spike_1wTrend
Hypothesis: TRIX (triple-smoothed EMA) momentum on 12h combined with weekly trend filter and volume spike
provides robust signals in both bull and bear markets. Weekly trend ensures we only trade with the higher
timeframe direction, reducing whipsaw. Volume spike confirms institutional participation. Target: 20-40
trades/year on 12h timeframe with strict entry conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def _ema(arr, period):
    """Calculate EMA with proper handling of initial values."""
    ema = np.full_like(arr, np.nan, dtype=float)
    if len(arr) < period:
        return ema
    k = 2.0 / (period + 1)
    ema[period-1] = np.mean(arr[:period])
    for i in range(period, len(arr)):
        ema[i] = arr[i] * k + ema[i-1] * (1 - k)
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # TRIX on 12h: triple EMA of percent change
    # TRIX = EMA(EMA(EMA(ROC, 12), 12), 12) where ROC = (close - close_prev)/close_prev * 100
    if len(close) < 2:
        return np.zeros(n)
    roc = np.zeros(n)
    roc[1:] = (close[1:] - close[:-1]) / close[:-1] * 100.0
    
    ema1 = _ema(roc, 12)
    ema2 = _ema(ema1, 12)
    ema3 = _ema(ema2, 12)
    trix = ema3  # Already the final smoothed value
    
    # Weekly trend filter: EMA34 on weekly close
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema34_1w = _ema(close_1w, 34)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume spike: current volume > 2.0 x 24-period average (24*12h = 12 days)
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 12*3)  # Ensure TRIX and volume MA ready
    
    for i in range(start_idx, n):
        if (np.isnan(trix[i]) or np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX turns positive with volume spike and weekly uptrend
            if (trix[i] > 0.0 and trix[i-1] <= 0.0 and vol_spike[i] and 
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: TRIX turns negative with volume spike and weekly downtrend
            elif (trix[i] < 0.0 and trix[i-1] >= 0.0 and vol_spike[i] and 
                  close[i] < ema34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TRIX turns negative or weekly trend turns down
            if (trix[i] < 0.0 and trix[i-1] >= 0.0) or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TRIX turns positive or weekly trend turns up
            if (trix[i] > 0.0 and trix[i-1] <= 0.0) or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_TRIX_Volume_Spike_1wTrend"
timeframe = "12h"
leverage = 1.0