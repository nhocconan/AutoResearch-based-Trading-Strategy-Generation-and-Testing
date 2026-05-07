#!/usr/bin/env python3
name = "12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSurge_v1"
timeframe = "12h"
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
    
    # 1w trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 1w EMA34 trend
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    trend_up = close > ema_34_1w_aligned
    trend_down = close < ema_34_1w_aligned
    
    # 1d Camarilla levels from previous 1d
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_prev_1d = df_1d['close'].values
    high_prev_1d = df_1d['high'].values
    low_prev_1d = df_1d['low'].values
    range_prev_1d = high_prev_1d - low_prev_1d
    # R1 and S1 levels (tighter)
    r1 = close_prev_1d + range_prev_1d * 1.1 / 12
    s1 = close_prev_1d - range_prev_1d * 1.1 / 12
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume surge: current volume > 2.0x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_surge = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 8  # ~4 days (8*12h) to reduce trade frequency
    
    start_idx = max(20, 1)  # Ensure enough data for volume and Camarilla
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine trend direction
        trending_up = trend_up[i]
        trending_down = trend_down[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: price breaks above R1 with volume surge in 1w uptrend
            if (close[i] > r1_aligned[i] and 
                trending_up and 
                vol_surge[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: price breaks below S1 with volume surge in 1w downtrend
            elif (close[i] < s1_aligned[i] and 
                  trending_down and 
                  vol_surge[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: price breaks below S1 or 1w trend changes to down
            if close[i] < s1_aligned[i] or not trending_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above R1 or 1w trend changes to up
            if close[i] > r1_aligned[i] or not trending_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Breakout at 1d Camarilla R1/S1 with volume surge and 1w trend filter works in both bull and bear markets.
# In bull markets: 1w trend up, breakouts above R1 capture continuation.
# In bear markets: 1w trend down, breakdowns below S1 capture continuation.
# Volume surge confirms institutional participation. Using R1/S1 (tighter than R2/S2) increases signal quality.
# Reduced cooldown to 8 bars (4 days) and position size 0.25 to target 50-150 trades over 4 years.
# 12h timeframe balances responsiveness with trade frequency control.