#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout + 1w trend filter + volume confirmation
# Camarilla levels from daily: R3/S3 for mean reversion, R4/S4 for breakout continuation
# Weekly trend filter: price > weekly EMA20 = bull trend, < = bear trend
# Volume confirmation: 6h volume > 1.5x daily average volume to avoid fakeouts
# Works in bull/bear: weekly trend aligns with higher timeframe direction
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "6h_1w_camarilla_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot levels (based on previous day)
    pivot_1d = (high_1d[:-1] + low_1d[:-1] + close_1d[:-1]) / 3.0
    range_1d = high_1d[:-1] - low_1d[:-1]
    
    # R3, S3, R4, S4 levels
    r3_1d = pivot_1d + range_1d * 1.1 / 2.0
    s3_1d = pivot_1d - range_1d * 1.1 / 2.0
    r4_1d = pivot_1d + range_1d * 1.1
    s4_1d = pivot_1d - range_1d * 1.1
    
    # Shift to align with current day (today's levels based on yesterday)
    r3_1d = np.concatenate([[np.nan], r3_1d])
    s3_1d = np.concatenate([[np.nan], s3_1d])
    r4_1d = np.concatenate([[np.nan], r4_1d])
    s4_1d = np.concatenate([[np.nan], s4_1d])
    
    # Calculate daily average volume (20-period)
    volume_1d = df_1d['volume'].values
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all HTF indicators to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(avg_volume_1d_aligned[i]) or np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x daily average volume
        volume_confirmed = volume[i] > 1.5 * avg_volume_1d_aligned[i]
        
        # Weekly trend: price > weekly EMA20 = bull, < = bear
        weekly_bull = close[i] > ema_20_1w_aligned[i]
        weekly_bear = close[i] < ema_20_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below S3 (mean reversion) OR weekly trend turns bear
            if close[i] < s3_1d_aligned[i] or weekly_bear:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above R3 (mean reversion) OR weekly trend turns bull
            if close[i] > r3_1d_aligned[i] or weekly_bull:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic based on weekly trend and Camarilla levels
            if weekly_bull:
                # In bull trend: look for breakout above R4 or mean reversion from S3
                if close[i] > r4_1d_aligned[i] and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < s3_1d_aligned[i] and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
            elif weekly_bear:
                # In bear trend: look for breakdown below S4 or mean reversion from R3
                if close[i] < s4_1d_aligned[i] and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
                elif close[i] > r3_1d_aligned[i] and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
    
    return signals