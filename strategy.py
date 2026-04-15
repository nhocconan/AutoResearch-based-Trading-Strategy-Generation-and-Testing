#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d volume + 1w trend filter
# Uses Elder Ray (bull/bear power) from 1d to measure trend strength, 
# with volume confirmation to filter false signals and weekly EMA for trend alignment.
# Works in both bull and bear markets by adapting to trend strength while avoiding whipsaw.
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h data (primary timeframe) for price action
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Load 1d data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Load 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate Elder Ray components (13-period EMA)
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema13_1d  # Bull Power: High - EMA13
    bear_power = low_1d - ema13_1d   # Bear Power: Low - EMA13
    
    # Volume average (20-period on 1d)
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # EMA50 on 1w for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Base position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(vol_avg_aligned[i]) or np.isnan(ema50_1w_aligned[i])):
            continue
        
        # Long entry: Bull Power > 0 (bullish momentum) + price above 6h EMA20 + volume spike + weekly uptrend
        if (bull_power_aligned[i] > 0 and 
            close[i] > pd.Series(close_6h).ewm(span=20, adjust=False, min_periods=20).mean().values[i] and
            volume[i] > 1.5 * vol_avg_aligned[i] and 
            close[i] > ema50_1w_aligned[i] and 
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Bear Power < 0 (bearish momentum) + price below 6h EMA20 + volume spike + weekly downtrend
        elif (bear_power_aligned[i] < 0 and 
              close[i] < pd.Series(close_6h).ewm(span=20, adjust=False, min_periods=20).mean().values[i] and
              volume[i] > 1.5 * vol_avg_aligned[i] and 
              close[i] < ema50_1w_aligned[i] and 
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or Elder Ray divergence (loss of momentum)
        elif position == 1 and bull_power_aligned[i] <= 0:
            position = 0
            signals[i] = 0.0
        elif position == -1 and bear_power_aligned[i] >= 0:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_1dVolume_1wEMA_Trend"
timeframe = "6h"
leverage = 1.0