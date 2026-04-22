#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
# Weekly pivot (from 1w) provides directional bias from higher timeframe structure.
# Donchian breakout captures momentum in direction of weekly trend.
# Volume confirmation (>1.5x 20-period average) filters false breakouts.
# Designed for 6h timeframe targeting 15-30 trades/year (~60-120 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for pivot points (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot points: P = (H + L + C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    
    # Align weekly pivots to 6h
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Donchian Channel (20) on 6h data
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or
            np.isnan(s1_1w_aligned[i]) or np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Donchian breakout above upper band + price above weekly R1 + volume confirmation
            if (close[i] > highest_20[i-1] and  # breakout above previous period's high
                close[i] > r1_1w_aligned[i] and   # price above weekly R1 (bullish bias)
                volume[i] > 1.5 * vol_avg_20[i]): # volume spike
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout below lower band + price below weekly S1 + volume confirmation
            elif (close[i] < lowest_20[i-1] and   # breakout below previous period's low
                  close[i] < s1_1w_aligned[i] and  # price below weekly S1 (bearish bias)
                  volume[i] > 1.5 * vol_avg_20[i]): # volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite pivot level or opposite Donchian band
            if position == 1:
                # Exit long: price returns to weekly S1 or below lower Donchian band
                if (close[i] < s1_1w_aligned[i] or 
                    close[i] < lowest_20[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price returns to weekly R1 or above upper Donchian band
                if (close[i] > r1_1w_aligned[i] or 
                    close[i] > highest_20[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivot_VolumeConfirm"
timeframe = "6h"
leverage = 1.0