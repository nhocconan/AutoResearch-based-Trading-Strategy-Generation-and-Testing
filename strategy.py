#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_Session
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation.
Enters long when price breaks above R1 with bullish 4h trend and volume spike.
Enters short when price breaks below S1 with bearish 4h trend and volume spike.
Exits when price reverses to opposite Camarilla level (S1/R1) or trend changes.
Uses 1h primary timeframe targeting 15-37 trades/year (60-150 total over 4 years).
Session filter (08-20 UTC) reduces noise. Works in bull/bear markets by aligning with 4h trend.
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
    
    # Get 4h data for trend and Camarilla pivot calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate previous 4h bar's Camarilla pivot levels (R1, S1)
    high_4h_prev = np.roll(df_4h['high'].values, 1)
    low_4h_prev = np.roll(df_4h['low'].values, 1)
    close_4h_prev = np.roll(df_4h['close'].values, 1)
    high_4h_prev[0] = np.nan
    low_4h_prev[0] = np.nan
    close_4h_prev[0] = np.nan
    
    # Camarilla pivot calculation
    pivot = (high_4h_prev + low_4h_prev + close_4h_prev) / 3.0
    range_4h = high_4h_prev - low_4h_prev
    r1 = pivot + (range_4h * 1.0 / 12.0)  # R1 level
    s1 = pivot - (range_4h * 1.0 / 12.0)  # S1 level
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    
    # Volume confirmation: volume > 2.0x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for EMA, 20 for volume MA, 1 for pivot)
    start_idx = max(50, 20, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema_50_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(volume_spike[i]) or not in_session[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        if position == 0:
            # Long: price breaks above R1 with 4h bullish trend and volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_50_aligned[i] and volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 with 4h bearish trend and volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_50_aligned[i] and volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: price closes below S1 OR 4h trend turns bearish
            if (close[i] < s1_aligned[i] or close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: price closes above R1 OR 4h trend turns bullish
            if (close[i] > r1_aligned[i] or close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0