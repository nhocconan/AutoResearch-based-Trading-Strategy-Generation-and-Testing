#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Supertrend (ATR=10, mult=3.0) for trend direction and
# 1d Camarilla R3/S3 levels for breakout entries with volume confirmation (>2.0x 20-bar avg volume).
# Long when: price breaks above 1d Camarilla R3, volume > 2.0x avg, and 12h Supertrend is bullish.
# Short when: price breaks below 1d Camarilla S3, volume > 2.0x avg, and 12h Supertrend is bearish.
# Exit when: price returns to 1d Camarilla midpoint (P) or touches opposite level (S3 for long exit, R3 for short exit).
# Uses discrete position sizing (0.25) to control risk. Target: 50-150 total trades over 4 years.
# Works in bull markets (breakouts continue with trend) and bear markets (breakdowns continue with trend).

name = "6h_Camarilla_R3S3_Breakout_12hSupertrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range for Camarilla calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - close_1d)
    tr3 = np.abs(low_1d - close_1d)
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Camarilla levels (based on previous day's close and range)
    camarilla_pivot = close_1d  # Pivot is previous close
    camarilla_range = high_1d - low_1d
    
    # R3 and S3 levels
    r3 = camarilla_pivot + camarilla_range * 1.1 / 4
    s3 = camarilla_pivot - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Get 12h data for Supertrend (ATR=10, mult=3.0) trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # Calculate 12h Supertrend
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # ATR calculation
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = high_12h[0] - low_12h[0]  # First bar: no previous close
    tr2[0] = np.abs(high_12h[0] - close_12h[0])  # First bar: use current close
    tr3[0] = np.abs(low_12h[0] - close_12h[0])   # First bar: use current close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (high_12h + low_12h) / 2
    upperband = hl2 + (3.0 * atr)
    lowerband = hl2 - (3.0 * atr)
    
    supertrend = np.zeros_like(close_12h)
    direction = np.ones_like(close_12h)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upperband[0]
    direction[0] = 1
    
    for i in range(1, len(close_12h)):
        if close_12h[i] > upperband[i-1]:
            direction[i] = 1
        elif close_12h[i] < lowerband[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1:
            supertrend[i] = max(lowerband[i], supertrend[i-1])
        else:
            supertrend[i] = min(upperband[i], supertrend[i-1])
    
    # Align Supertrend direction to 6h timeframe (1 for bullish, -1 for bearish)
    supertrend_direction_aligned = align_htf_to_ltf(prices, df_12h, direction)
    
    # Calculate 6h volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(pivot_aligned[i]) or
            np.isnan(supertrend_direction_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Trend filter: 12h Supertrend direction
        bullish_bias = supertrend_direction_aligned[i] == 1
        bearish_bias = supertrend_direction_aligned[i] == -1
        
        # Camarilla breakout conditions
        long_breakout = close[i] > r3_aligned[i]
        short_breakout = close[i] < s3_aligned[i]
        
        # Exit conditions: return to pivot or opposite level touched
        long_exit = close[i] < pivot_aligned[i]
        short_exit = close[i] > pivot_aligned[i]
        
        # Entry conditions
        long_entry = long_breakout and vol_confirm and bullish_bias
        short_entry = short_breakout and vol_confirm and bearish_bias
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals