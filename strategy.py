#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Weighted
Hypothesis: Trading breakouts at inner Camarilla levels (R1/S1) from daily pivot with volume confirmation and 1-day trend filter captures strong momentum moves in both bull and bear markets. Uses volume-weighted price for more robust breakout detection. Timeframe: 12h balances trade frequency and signal quality for 1-day trend alignment, targeting 12-37 trades/year with low fee drift.
"""

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Weighted"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for current day (using previous day's OHLC)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Calculate Camarilla levels (focusing on R1/S1 for breakout)
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Get 1d data for EMA trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 12h data for signal generation
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume-weighted price: (high + low + close) / 3
    vwp = (high + low + close) / 3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 1d EMA34 (34 periods)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price vs 1d EMA34
        uptrend_1d = close[i] > ema34_1d_aligned[i]
        downtrend_1d = close[i] < ema34_1d_aligned[i]
        
        # Volume filter: current volume > 1.5x 20-period EMA
        vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
        volume_filter = volume[i] > vol_ema20[i] * 1.5
        
        if position == 0:
            # Long breakout: volume-weighted price breaks above R1 with volume and uptrend
            if vwp[i] > R1_aligned[i] and uptrend_1d and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short breakout: volume-weighted price breaks below S1 with volume and downtrend
            elif vwp[i] < S1_aligned[i] and downtrend_1d and volume_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches midpoint between R1 and S1 or trend fails
            midpoint = (R1_aligned[i] + S1_aligned[i]) / 2
            if low[i] <= midpoint or not uptrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches midpoint between R1 and S1 or trend fails
            midpoint = (R1_aligned[i] + S1_aligned[i]) / 2
            if high[i] >= midpoint or not downtrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals