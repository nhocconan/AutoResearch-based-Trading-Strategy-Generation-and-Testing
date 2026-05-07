#!/usr/bin/env python3
"""
1D_R1S1_Breakout_1WTrend_Volume
Hypothesis: Daily price breaks above/below weekly Camarilla R1/S1 levels with weekly EMA34 trend confirmation and volume spike.
Weekly trend filter avoids counter-trend trades, reducing whipsaws in ranging markets.
Volume confirmation ensures breakout strength.
Targets 7-25 trades/year on 1d timeframe to minimize fee drag.
"""
name = "1D_R1S1_Breakout_1WTrend_Volume"
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
    
    # Get weekly data for Camarilla levels, EMA trend, and volume average
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels (R1, S1)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pivot = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    r1 = pivot + (range_1w * 1.1 / 6)  # R1 level
    s1 = pivot - (range_1w * 1.1 / 6)  # S1 level
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Calculate weekly EMA34 for trend direction
    close_1w_series = pd.Series(df_1w['close'])
    ema_34 = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Volume filter: current 1d volume > 1.5 x 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(34, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 10 days between trades to reduce frequency
            if bars_since_exit < 10:
                continue
                
            # Long: price breaks above R1 with weekly EMA34 uptrend and volume spike
            if (close[i] > r1_aligned[i] and close[i-1] <= r1_aligned[i-1] and 
                close[i] > ema_34_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: price breaks below S1 with weekly EMA34 downtrend and volume spike
            elif (close[i] < s1_aligned[i] and close[i-1] >= s1_aligned[i-1] and 
                  close[i] < ema_34_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite EMA34 side (trend reversal)
            if position == 1 and close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals