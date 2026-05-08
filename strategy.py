#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyDonchian_Breakout_VolumeTrend"
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
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Calculate 1d EMA(50) for trend filter
    ema50_1d = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema50_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        ema50 = ema50_1d[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: break above weekly Donchian high + above EMA50 + volume spike
            if (close[i] > upper and 
                close[i] > ema50 and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: break below weekly Donchian low + below EMA50 + volume spike
            elif (close[i] < lower and 
                  close[i] < ema50 and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: close below weekly Donchian low OR below EMA50
            if (close[i] < lower or close[i] < ema50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: close above weekly Donchian high OR above EMA50
            if (close[i] > upper or close[i] > ema50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals