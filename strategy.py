#!/usr/bin/env python3
"""
6h_Liquidity_Sweep_Retest_With_Volume
Hypothesis: In 6h timeframe, price often sweeps liquidity (equal highs/lows) before reversing.
We detect equal highs/lows within 0.2% tolerance, wait for retest of the swept level with
volume confirmation, and trade in the direction of the reversal. Uses 1d trend filter to
avoid counter-trend trades. Designed for low frequency (~20-40 trades/year) to minimize
fee drag. Works in both bull and bear markets as liquidity sweeps occur in all regimes.
"""

name = "6h_Liquidity_Sweep_Retest_With_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d trend filter: EMA(50) on close
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Detect equal highs/lows (liquidity pools) with 0.2% tolerance
    equal_high = np.zeros(n, dtype=bool)
    equal_low = np.zeros(n, dtype=bool)
    
    for i in range(5, n):
        # Check for equal high (within 0.2%)
        for j in range(i-4, i):
            if abs(high[i] - high[j]) / high[i] < 0.002:
                equal_high[i] = True
                break
        # Check for equal low (within 0.2%)
        for j in range(i-4, i):
            if abs(low[i] - low[j]) / low[i] < 0.002:
                equal_low[i] = True
                break
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(5, n):
        if position == 0:
            # LONG: Price swept equal low (liquidity grab), now retesting that level with volume
            if equal_low[i] and low[i] <= high[i-1]:  # Price closed above the swept low
                # Additional confirmation: price above 1d EMA50 (uptrend bias)
                if close[i] > ema50_1d_aligned[i] and volume_filter[i]:
                    signals[i] = 0.25
                    position = 1
            # SHORT: Price swept equal high (liquidity grab), now retesting that level with volume
            elif equal_high[i] and high[i] >= low[i-1]:  # Price closed below the swept high
                # Additional confirmation: price below 1d EMA50 (downtrend bias)
                if close[i] < ema50_1d_aligned[i] and volume_filter[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below the swept low level OR volume drops
            if low[i] < low[i-1] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above the swept high level OR volume drops
            if high[i] > high[i-1] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals