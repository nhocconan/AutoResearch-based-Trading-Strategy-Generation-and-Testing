#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_Session
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA20 trend filter, volume spike (>2.0x median), and session filter (08-20 UTC). Enters long when price breaks above R1 with volume confirmation, bullish 4h trend, and active session. Enters short when price breaks below S1 with volume confirmation, bearish 4h trend, and active session. Exits on opposite breakout. Uses discrete position sizing (0.20) to minimize churn. Target: 60-150 total trades over 4 years. Works in both bull and bear markets by following 4h trend filter and avoiding low-volume, off-session noise.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Calculate Camarilla levels for 1h (based on previous 1h bar)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 2:
        return np.zeros(n)
    
    h_1h = df_1h['high'].values
    l_1h = df_1h['low'].values
    c_1h = df_1h['close'].values
    
    # Previous bar's values for level calculation (to avoid look-ahead)
    h_1h_prev = np.roll(h_1h, 1)
    l_1h_prev = np.roll(l_1h, 1)
    c_1h_prev = np.roll(c_1h, 1)
    h_1h_prev[0] = np.nan
    l_1h_prev[0] = np.nan
    c_1h_prev[0] = np.nan
    
    # Calculate Camarilla R1 and S1 levels
    rng_1h = h_1h_prev - l_1h_prev
    r1_1h = c_1h_prev + (rng_1h * 1.1 / 12)
    s1_1h = c_1h_prev - (rng_1h * 1.1 / 12)
    
    # Align to 1h primary timeframe
    r1_1h_aligned = align_htf_to_ltf(prices, df_1h, r1_1h)
    s1_1h_aligned = align_htf_to_ltf(prices, df_1h, s1_1h)
    
    # Volume confirmation: volume > 2.0x 20-period median
    volume_series = pd.Series(volume)
    vol_median = volume_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (2.0 * vol_median)
    
    # Load 4h data for HTF trend filter (EMA20)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Session filter: 08-20 UTC (inclusive)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.20
    
    # Start after warmup (need 20-period volume median, 20-period EMA)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_1h_aligned[i]) or np.isnan(s1_1h_aligned[i]) or 
            np.isnan(vol_median[i]) or np.isnan(ema_20_4h_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Only trade during active session (08-20 UTC)
        if not in_session[i]:
            # Hold current position outside session
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: price breaks above R1 + volume spike + bullish 4h trend + in session
        if close[i] > r1_1h_aligned[i] and volume_spike[i] and close[i] > ema_20_4h_aligned[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: price breaks below S1 + volume spike + bearish 4h trend + in session
        elif close[i] < s1_1h_aligned[i] and volume_spike[i] and close[i] < ema_20_4h_aligned[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: opposite breakout (price returns to the other level)
        elif position == 1 and close[i] < s1_1h_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > r1_1h_aligned[i]:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0