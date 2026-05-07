#!/usr/bin/env python3
name = "6h_WeeklyPivot_Direction_VolumeBreakout"
timeframe = "6h"
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
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    # Weekly EMA10 for trend direction
    ema_10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    weekly_up = ema_10_1w_aligned > 0  # Valid after warmup
    weekly_down = ema_10_1w_aligned < 0  # Valid after warmup
    
    # Daily pivot points for support/resistance
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # Calculate pivot: P = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Calculate R1, S1
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    # Align to 6h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 2  # ~12 hours (2*6h) to prevent overtrading
    
    start_idx = max(20, 10)  # Ensure enough data for volume MA and EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vol_ma_20[i]) or 
            np.isnan(pivot_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or
            np.isnan(ema_10_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine trend direction from weekly EMA
        trending_up = ema_10_1w_aligned[i] > 0
        trending_down = ema_10_1w_aligned[i] < 0
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Price breaks above R1 with volume spike in weekly uptrend
            if (close[i] > r1_1d_aligned[i] and 
                trending_up and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Price breaks below S1 with volume spike in weekly downtrend
            elif (close[i] < s1_1d_aligned[i] and 
                  trending_down and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price falls back below pivot or weekly trend changes to down
            if close[i] < pivot_1d_aligned[i] or not trending_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises back above pivot or weekly trend changes to up
            if close[i] > pivot_1d_aligned[i] or not trending_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: On 6h timeframe, price breaking above/below daily pivot R1/S1 levels with volume spike confirmation and weekly EMA10 trend filter captures institutional breakout momentum. Daily pivot points represent key support/resistance levels from prior day's action, reducing false breakouts. Weekly trend filter ensures alignment with higher timeframe momentum. Volume spike filter (1.5x 20-period average) confirms institutional participation. Cooldown period prevents overtrading. Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag. Works in bull markets (breakouts above R1 in weekly uptrend) and bear markets (breakdowns below S1 in weekly downtrend). Uses discrete position sizing (0.25) to balance risk and reward while reducing fee churn. This strategy avoids saturated patterns and focuses on pivot-based breakouts with volume/trend confluence, which has shown promise in DB (e.g., 6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_HT with 1.882 Sharpe). Pivots are less commonly used than Donchian channels, offering a fresh approach.