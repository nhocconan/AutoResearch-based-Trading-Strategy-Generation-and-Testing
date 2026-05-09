#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Donchian20_WeeklyPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly pivot and levels from previous week's OHLC
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    prev_close_1w[0] = np.nan
    
    prev_weekly_range = prev_high_1w - prev_low_1w
    pivot = (prev_high_1w + prev_low_1w + prev_close_1w) / 3
    r1 = pivot + 1.1 * prev_weekly_range / 6
    s1 = pivot - 1.1 * prev_weekly_range / 6
    
    # Align weekly pivot levels to 6h
    r1_6h = align_htf_to_ltf(prices, df_1w, r1)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1)
    
    # 20-period Donchian channel on 6h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.8 x 20-period average (adjusted for 6h)
        vol_spike = volume[i] > vol_avg[i] * 1.8
        
        if position == 0:
            # Long: Break above Donchian high with weekly bullish bias and volume spike
            if close[i] > donchian_high[i] and close[i] > r1_6h[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low with weekly bearish bias and volume spike
            elif close[i] < donchian_low[i] and close[i] < s1_6h[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below Donchian low or weekly S1
            if close[i] < donchian_low[i] or close[i] < s1_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above Donchian high or weekly R1
            if close[i] > donchian_high[i] or close[i] > r1_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals