#!/usr/bin/env python3
name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Data for trend filter ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 4h EMA20 for trend direction
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # === Daily data for Camarilla levels ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's values for Camarilla calculation
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    range_1d = prev_high_1d - prev_low_1d
    
    # Camarilla R1 and S1 levels
    r1 = prev_close_1d + (range_1d * 1.1 / 12)
    s1 = prev_close_1d - (range_1d * 1.1 / 12)
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === Volume spike detection (1h) ===
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values  # ~1 day of 1h bars
    volume_spike = volume > (vol_ma * 1.5)
    
    # === Session filter: 08-20 UTC ===
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 20)  # Ensure enough data for volume MA and EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema20_4h_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Only trade during active session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + volume spike + 4h trend up
            if (close[i] > r1_aligned[i] and 
                volume_spike[i] and
                close[i] > ema20_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 + volume spike + 4h trend down
            elif (close[i] < s1_aligned[i] and 
                  volume_spike[i] and
                  close[i] < ema20_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below S1 or trend breaks
            if close[i] < s1_aligned[i] or close[i] < ema20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above R1 or trend breaks
            if close[i] > r1_aligned[i] or close[i] > ema20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals