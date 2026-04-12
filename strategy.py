#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla pivot breakout with 1d volume confirmation
    # Camarilla levels (R3/S3, R4/S4) act as institutional support/resistance
    # Breakout above R4 or below S4 with volume confirmation = continuation signal
    # Fade at R3/S3 with volume exhaustion = mean reversion signal
    # Works in bull/bear by adapting to pivot structure and volume context
    # Target: 12-37 trades/year per symbol.
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = C + (Range * 1.1/2)
    # R3 = C + (Range * 1.1/4)
    # S3 = C - (Range * 1.1/4)
    # S4 = C - (Range * 1.1/2)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r4_1d = close_1d + (range_1d * 1.1 / 2.0)
    r3_1d = close_1d + (range_1d * 1.1 / 4.0)
    s3_1d = close_1d - (range_1d * 1.1 / 4.0)
    s4_1d = close_1d - (range_1d * 1.1 / 2.0)
    
    # Align 1d Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_20 = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        if i == 19:
            vol_ma_20[i] = np.mean(volume_1d[i-19:i+1])
        else:
            vol_ma_20[i] = (vol_ma_20[i-1] * 19 + volume_1d[i]) / 20
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x daily average volume
        # Need to approximate 6h volume relative to daily volume
        # Use volume ratio: if current volume is unusually high for the time of day
        vol_ratio = volume[i] / (vol_ma_20_aligned[i] / 4.0)  # 4x 6h bars per day approx
        
        # Breakout signals: price breaks R4/S4 with volume expansion
        breakout_long = (close[i] > r4_aligned[i]) and (vol_ratio > 1.5)
        breakout_short = (close[i] < s4_aligned[i]) and (vol_ratio > 1.5)
        
        # Mean reversion fade signals: price rejects R3/S3 with volume exhaustion
        fade_long = (close[i] < r3_aligned[i]) and (close[i-1] >= r3_aligned[i-1]) and (vol_ratio < 0.7)
        fade_short = (close[i] > s3_aligned[i]) and (close[i-1] <= s3_aligned[i-1]) and (vol_ratio < 0.7)
        
        # Exit conditions: return to pivot or opposite extreme
        long_exit = (close[i] < pivot_1d[-1] if len(pivot_1d) > 0 else close[i] < close[i]) or \
                    (position == 1 and close[i] < s3_aligned[i])
        short_exit = (close[i] > pivot_1d[-1] if len(pivot_1d) > 0 else close[i] > close[i]) or \
                     (position == -1 and close[i] > r3_aligned[i])
        
        # Simplified exits: use aligned pivot
        pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
        if not np.isnan(pivot_aligned[i]):
            long_exit = (close[i] < pivot_aligned[i]) or (position == 1 and close[i] < s3_aligned[i])
            short_exit = (close[i] > pivot_aligned[i]) or (position == -1 and close[i] > r3_aligned[i])
        
        if (breakout_long or fade_long) and position != 1:
            position = 1
            signals[i] = 0.25
        elif (breakout_short or fade_short) and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_camarilla_pivot_breakout_v1"
timeframe = "6h"
leverage = 1.0