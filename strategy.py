#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeSpike
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and 1d volume spike confirmation.
Uses higher timeframes for signal direction (4h trend, 1d volume regime) and 1h only for precise entry timing.
Session filter (08-20 UTC) reduces noise trades. Fixed size 0.20 to minimize fee churn.
Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years).
Works in bull/bear by following 4h trend and avoiding low-volume periods.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for volume spike confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume MA for spike detection
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate previous day's Camarilla pivot levels (R1, S1) from 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift to get previous day's HLC (avoid look-ahead)
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    close_1d_prev[0] = np.nan
    
    # Camarilla pivot calculation
    pivot = (high_1d_prev + low_1d_prev + close_1d_prev) / 3.0
    range_1d = high_1d_prev - low_1d_prev
    r1 = pivot + (range_1d * 1.0 / 12.0)  # R1 level
    s1 = pivot - (range_1d * 1.0 / 12.0)  # S1 level
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 4h EMA, 20 for 1d volume MA, 1 for pivot)
    start_idx = max(50, 20, 1)
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Volume spike condition: current 1h volume > 2.0x 20-day volume MA
        volume_spike = volume[i] > (vol_ma_20_1d_aligned[i] * 2.0)
        
        if position == 0:
            # Long: price breaks above R1 with 4h bullish trend and volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_50_4h_aligned[i] and volume_spike):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 with 4h bearish trend and volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_50_4h_aligned[i] and volume_spike):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: price closes below S1 OR 4h trend turns bearish
            if (close[i] < s1_aligned[i] or close[i] < ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: price closes above R1 OR 4h trend turns bullish
            if (close[i] > r1_aligned[i] or close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeSpike"
timeframe = "1h"
leverage = 1.0