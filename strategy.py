#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeConfirmation
Hypothesis: 1d Camarilla R1/S1 breakout with 1w EMA50 trend filter and volume confirmation.
Enters long when price breaks above R1 with bullish 1w trend and volume spike.
Enters short when price breaks below S1 with bearish 1w trend and volume spike.
Exits when price reverses to opposite Camarilla level or trend changes.
Uses 1d primary timeframe to target 7-25 trades/year (30-100 total over 4 years).
Works in bull/bear markets by aligning with 1w trend to avoid counter-trend trades.
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
    
    # Get 1w data for trend and Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate previous 1w bar's Camarilla pivot levels (R1, S1)
    # Need HLC from previous 1w bar to avoid look-ahead
    high_1w_prev = np.roll(df_1w['high'].values, 1)
    low_1w_prev = np.roll(df_1w['low'].values, 1)
    close_1w_prev = np.roll(df_1w['close'].values, 1)
    # First value will be invalid (rolled from last), set to nan
    high_1w_prev[0] = np.nan
    low_1w_prev[0] = np.nan
    close_1w_prev[0] = np.nan
    
    # Camarilla pivot calculation
    pivot = (high_1w_prev + low_1w_prev + close_1w_prev) / 3.0
    range_1w = high_1w_prev - low_1w_prev
    r1 = pivot + (range_1w * 1.0 / 12.0)  # R1 level
    s1 = pivot - (range_1w * 1.0 / 12.0)  # S1 level
    
    # Align Camarilla levels to 1d timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Volume confirmation: volume > 2.0x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for EMA, 20 for volume MA, 1 for pivot)
    start_idx = max(50, 20, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price breaks above R1 with 1w bullish trend and volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_50_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with 1w bearish trend and volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_50_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below S1 OR 1w trend turns bearish
            if (close[i] < s1_aligned[i] or close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above R1 OR 1w trend turns bullish
            if (close[i] > r1_aligned[i] or close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0