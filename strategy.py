#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Donchian20_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    # Calculate weekly EMA(20) for trend direction
    close_weekly = df_weekly['close'].values
    ema20_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema20_weekly)
    
    # Calculate weekly Donchian(20) breakout levels
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    donch_high = pd.Series(high_weekly).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_weekly).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_weekly, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_weekly, donch_low)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema20_weekly_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema20_weekly_val = ema20_weekly_aligned[i]
        upper = donch_high_aligned[i]
        lower = donch_low_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above weekly Donchian high + uptrend + volume spike
            if (close[i] > upper and 
                close[i] > ema20_weekly_val and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly Donchian low + downtrend + volume spike
            elif (close[i] < lower and 
                  close[i] < ema20_weekly_val and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below weekly Donchian low OR trend turns down
            if (close[i] < lower or close[i] < ema20_weekly_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above weekly Donchian high OR trend turns up
            if (close[i] > upper or close[i] > ema20_weekly_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals