# 1:6s
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot (R3/S3) breakout with 4h trend filter and volume spike confirmation.
# Long when price breaks above R3 (resistance level 3) AND 4h EMA50 rising AND volume > 1.5x 20-period average.
# Short when price breaks below S3 (support level 3) AND 4h EMA50 falling AND volume > 1.5x 20-period average.
# Exit when price crosses back to P (pivot point).
# Uses Camarilla pivots for intraday levels, 4h EMA for trend alignment, volume for confirmation.
# Target: 60-150 total trades over 4 years (15-37/year) with session filter (08-20 UTC).

name = "1h_Camarilla_R3S3_4hEMA50_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for Camarilla calculation and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivots from 4h data
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Pivot point (P)
    pivot_4h = (high_4h + low_4h + close_4h) / 3
    # Resistance and support levels
    range_4h = high_4h - low_4h
    r3_4h = close_4h + (range_4h * 1.1 / 2)
    s3_4h = close_4h - (range_4h * 1.1 / 2)
    
    # Align Camarilla levels to 1h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_4h, pivot_4h)
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    
    # 4h EMA50 for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 4h EMA50 direction
    ema50_rising = np.zeros_like(ema50_4h_aligned, dtype=bool)
    ema50_falling = np.zeros_like(ema50_4h_aligned, dtype=bool)
    ema50_rising[1:] = ema50_4h_aligned[1:] > ema50_4h_aligned[:-1]
    ema50_falling[1:] = ema50_4h_aligned[1:] < ema50_4h_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Sufficient warmup for EMA50 and pivots
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(ema50_rising[i]) or 
            np.isnan(ema50_falling[i]) or np.isnan(volume_filter[i]) or np.isnan(session_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3, 4h EMA50 rising, volume filter
            long_cond = (close[i] > r3_aligned[i]) and ema50_rising[i] and volume_filter[i]
            # Short conditions: price breaks below S3, 4h EMA50 falling, volume filter
            short_cond = (close[i] < s3_aligned[i]) and ema50_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price crosses back below pivot
            if close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price crosses back above pivot
            if close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals