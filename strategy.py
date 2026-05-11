#!/usr/bin/env python3
"""
4h_4hDonchian20_1dTrend_Volume
Hypothesis: Breakouts at 4h Donchian(20) high/low on 4h timeframe with 1d trend filter and volume confirmation.
Donchian breakouts capture momentum moves. The 1d trend filter ensures alignment with higher timeframe bias,
reducing false signals. Volume confirmation adds conviction. Works in bull/bear markets by following 1d trend.
"""

name = "4h_4hDonchian20_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Donchian Channel (20-period) ===
    # Highest high of last 20 bars (including current)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lowest low of last 20 bars (including current)
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1d Trend Filter (EMA50) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === Volume Filter (1.5x 20-period EMA on 4h) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers Donchian and EMA calculations)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price closes above Donchian high with uptrend and volume
            if (close[i] > highest_high[i] and 
                close[i] > ema50_1d_aligned[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price closes below Donchian low with downtrend and volume
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below Donchian low (mean reversion)
            if close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price closes above Donchian high (mean reversion)
            if close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals