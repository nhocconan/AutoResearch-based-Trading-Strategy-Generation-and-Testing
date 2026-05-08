#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian channel breakout with weekly trend filter and volume confirmation
# Long when price breaks above 20-day high + weekly EMA(21) uptrend + volume spike
# Short when price breaks below 20-day low + weekly EMA(21) downtrend + volume spike
# Donchian breakouts capture momentum; weekly trend filter ensures alignment with higher timeframe
# Volume spike confirms institutional participation; targets 30-100 total trades over 4 years
# Uses daily timeframe as primary, weekly as higher timeframe for trend filter

name = "1d_Donchian20_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate weekly EMA(21) for trend filter
    weekly_close = df_1w['close'].values
    ema21_1w = pd.Series(weekly_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Calculate Donchian channels (20-period) on daily data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup for Donchian and EMA calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema21_1w_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema21_1w_val = ema21_1w_aligned[i]
        upper = high_20[i]
        lower = low_20[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above 20-day high + weekly uptrend + volume spike
            if close[i] > upper and close[i] > ema21_1w_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 20-day low + weekly downtrend + volume spike
            elif close[i] < lower and close[i] < ema21_1w_val and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below 20-day low OR weekly trend turns down
            if close[i] < lower or close[i] < ema21_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above 20-day high OR weekly trend turns up
            if close[i] > upper or close[i] > ema21_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals