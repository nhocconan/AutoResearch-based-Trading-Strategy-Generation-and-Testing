#!/usr/bin/env python3
"""
12h_Donchian_Breakout_1dTrend_Volume
Hypothesis: Use 12h Donchian channel breakout (20-period high/low) for entry, filtered by 1d EMA trend (trend-following) and volume spike (volume > 1.5x 20-period average). Exit on opposite Donchian breakout or trend reversal. Designed for 12h timeframe to limit trades (target: 50-150 total over 4 years) and avoid fee drag. Works in bull markets (breakouts capture momentum) and bear markets (trend filter prevents whipsaw, volume confirms strength).
"""

name = "12h_Donchian_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period volume average for volume spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h Donchian channels (20-period)
    # We need at least 20 periods for Donchian calculation
    lookback = 20
    if n < lookback:
        return np.zeros(n)
    
    # Calculate rolling max/min for Donchian channels
    high_max = np.full(n, np.nan)
    low_min = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        high_max[i] = np.max(high[i - lookback + 1:i + 1])
        low_min[i] = np.min(low[i - lookback + 1:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(high_max[i]) or np.isnan(low_min[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5x 20-period average
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above Donchian upper band + uptrend (price > EMA50) + volume spike
            if close[i] > high_max[i] and close[i] > ema_50_1d_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower band + downtrend (price < EMA50) + volume spike
            elif close[i] < low_min[i] and close[i] < ema_50_1d_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian lower band or trend reversal (price < EMA50)
            if close[i] < low_min[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian upper band or trend reversal (price > EMA50)
            if close[i] > high_max[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals