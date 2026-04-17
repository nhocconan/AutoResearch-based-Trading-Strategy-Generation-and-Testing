#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_R1_S1_Breakout_Volume_Spike_v1
1-day Camarilla Pivot R1/S1 breakout with volume spike confirmation.
Long on break above R1 with volume > 1.5x avg volume(20).
Short on break below S1 with volume > 1.5x avg volume(20).
Exit when price closes back inside H3-L3 range.
Uses 1-week trend filter: only long when price > weekly EMA200, only short when price < weekly EMA200.
Target: 10-30 trades per year (~40-120 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Daily Camarilla Pivot Levels (based on previous day) ===
    # Calculate for each day using previous day's OHLC
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    
    # First day has no previous day
    prev_close[0] = close[0]
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla levels
    R1 = pivot + (range_val * 1.1 / 12)
    S1 = pivot - (range_val * 1.1 / 12)
    R3 = pivot + (range_val * 1.1 / 4)
    S3 = pivot - (range_val * 1.1 / 4)
    H3 = pivot + (range_val * 1.1 / 2)
    L3 = pivot - (range_val * 1.1 / 2)
    
    # === Volume Spike: volume > 1.5x 20-day average ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    # === 1-week EMA200 for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or 
            np.isnan(H3[i]) or np.isnan(L3[i]) or
            np.isnan(ema_200_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: break above R1 with volume spike and price above weekly EMA200
            if (close[i] > R1[i] and 
                vol_spike[i] and 
                close[i] > ema_200_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: break below S1 with volume spike and price below weekly EMA200
            elif (close[i] < S1[i] and 
                  vol_spike[i] and 
                  close[i] < ema_200_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price closes back inside H3-L3 range
            if L3[i] <= close[i] <= H3[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price closes back inside H3-L3 range
            if L3[i] <= close[i] <= H3[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_Pivot_R1_S1_Breakout_Volume_Spike_v1"
timeframe = "1d"
leverage = 1.0