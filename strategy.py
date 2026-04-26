#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeConfirm_v1
Hypothesis: Daily Camarilla R1/S1 breakout with weekly EMA50 trend filter and volume confirmation.
Only long when price breaks above R1 and weekly trend is up, short when price breaks below S1 and weekly trend is down.
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Designed for 30-100 total trades over 4 years (7-25/year).
Works in both bull and bear markets by combining price structure (Camarilla) with trend (weekly EMA) and volume filters.
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
    
    # Calculate Camarilla levels for daily timeframe (using previous day's range)
    # Camarilla R1 = close + (high - low) * 1.1/12
    # Camarilla S1 = close - (high - low) * 1.1/12
    # We need to use previous day's OHLC to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_high[1]  # fill first value
    prev_low[0] = prev_low[1]
    prev_close[0] = prev_close[1]
    
    daily_range = prev_high - prev_low
    camarilla_multiplier = 1.1 / 12
    r1 = prev_close + daily_range * camarilla_multiplier
    s1 = prev_close - daily_range * camarilla_multiplier
    
    # Load 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 1.5 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Discrete position sizing
        base_size = 0.25
        
        # Long logic: price breaks above R1 + weekly trend up + volume spike
        if close[i] > r1[i] and close[i] > ema_50_1w_aligned[i] and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: price breaks below S1 + weekly trend down + volume spike
        elif close[i] < s1[i] and close[i] < ema_50_1w_aligned[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit conditions: price returns to Camarilla H3/L3 levels or loss of volume confirmation
        elif position == 1 and (close[i] < (prev_close[i] + daily_range * 1.1/6) or not volume_spike[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > (prev_close[i] - daily_range * 1.1/6) or not volume_spike[i]):
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

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0