#!/usr/bin/env python3
"""
1D_Weekly_Trend_Filter_Camarilla_R1S1_Breakout
Hypothesis: Daily price breaks above/below weekly Camarilla R1/S1 levels with weekly EMA20 trend confirmation and volume spike.
Works in bull/bear markets: Weekly EMA filter captures major trend direction, weekly Camarilla levels act as strong support/resistance,
volume confirmation validates breakout strength. Targets 8-20 trades/year to minimize fee drag on 1d timeframe.
"""
name = "1D_Weekly_Trend_Filter_Camarilla_R1S1_Breakout"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend and Camarilla levels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend direction
    close_weekly = df_weekly['close'].values
    ema_20 = pd.Series(close_weekly).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_weekly, ema_20)
    
    # Calculate weekly Camarilla levels (R1, S1)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly_for_pivot = df_weekly['close'].values
    pivot = (high_weekly + low_weekly + close_weekly_for_pivot) / 3
    range_weekly = high_weekly - low_weekly
    r1 = pivot + (range_weekly * 1.1 / 6)  # R1 level
    s1 = pivot - (range_weekly * 1.1 / 6)  # S1 level
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    
    # Volume filter: current daily volume > 1.5 x 20-day average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(20, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 60 bars between trades (60 days on 1d TF) to reduce frequency
            if bars_since_exit < 60:
                continue
                
            # Long: price breaks above R1 with weekly EMA20 uptrend and volume spike
            if (close[i] > r1_aligned[i] and close[i-1] <= r1_aligned[i-1] and 
                close[i] > ema_20_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: price breaks below S1 with weekly EMA20 downtrend and volume spike
            elif (close[i] < s1_aligned[i] and close[i-1] >= s1_aligned[i-1] and 
                  close[i] < ema_20_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite weekly EMA20 side (trend reversal)
            if position == 1 and close[i] < ema_20_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and close[i] > ema_20_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals