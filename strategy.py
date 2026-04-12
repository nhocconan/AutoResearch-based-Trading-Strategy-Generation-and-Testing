#!/usr/bin/env python3
"""
12h_1d_keltner_volatility_breakout
Hypothesis: 12-hour strategy using Keltner Channel breakouts with 1-day trend filter and volume confirmation.
Works in bull/bear by requiring Keltner breakouts aligned with daily trend, using volatility to filter entries.
Target: 12-30 trades/year (48-120 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Daily EMA100 for trend filter
    ema100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    # 12h Keltner Channel (20-period EMA, 2*ATR)
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    upper = ema20 + 2 * atr
    lower = ema20 - 2 * atr
    
    # Volume confirmation: volume > 1.5x 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema100_1d_aligned[i]) or np.isnan(ema20[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price above EMA100 (uptrend) AND breaks above upper Keltner with volume
        if (close[i] > ema100_1d_aligned[i] and close[i] > upper[i] and vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price below EMA100 (downtrend) AND breaks below lower Keltner with volume
        elif (close[i] < ema100_1d_aligned[i] and close[i] < lower[i] and vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal or price crosses back to EMA20
        elif position == 1 and close[i] < ema20[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > ema20[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_keltner_volatility_breakout"
timeframe = "12h"
leverage = 1.0