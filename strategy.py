#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_R1_S1_Breakout_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend direction
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate weekly Camarilla levels from previous week
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_shift = df_1w['close'].shift(1).values
    high_1w_shift = df_1w['high'].shift(1).values
    low_1w_shift = df_1w['low'].shift(1).values
    
    # Calculate pivot and Camarilla levels using previous week's data
    pivot = (high_1w_shift + low_1w_shift + close_1w_shift) / 3
    range_ = high_1w_shift - low_1w_shift
    r1 = close_1w_shift + (range_ * 1.1 / 12)
    s1 = close_1w_shift - (range_ * 1.1 / 12)
    
    # Align Camarilla levels to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1w_val = ema34_1w_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above S1 + uptrend + volume spike
            if (close[i] > s1_val and 
                close[i] > ema34_1w_val and 
                vol_spike):
                signals[i] = 0.30
                position = 1
            # Enter short: price breaks below R1 + downtrend + volume spike
            elif (close[i] < r1_val and 
                  close[i] < ema34_1w_val and 
                  vol_spike):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price breaks below S1 OR trend turns down
            if (close[i] < s1_val or close[i] < ema34_1w_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price breaks above R1 OR trend turns up
            if (close[i] > r1_val or close[i] > ema34_1w_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals