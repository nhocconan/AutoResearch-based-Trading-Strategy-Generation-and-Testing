#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams Alligator (Jaw/Teeth/Lips) for trend direction and 1d weekly pivot levels for breakout entries.
# Long when: price breaks above 1d weekly R3 with Alligator bullish alignment (Lips > Teeth > Jaw) AND volume > 1.5x 20-bar average
# Short when: price breaks below 1d weekly S3 with Alligator bearish alignment (Lips < Teeth < Jaw) AND volume > 1.5x 20-bar average
# Exit via ATR(21) trailing stop: long exit when price < highest_high_since_entry - 3.0 * ATR
#                      short exit when price > lowest_low_since_entry + 3.0 * ATR
# Uses 1d Williams Alligator for smooth trend filtering (reduces whipsaw), 1d weekly pivot R3/S3 for precise breakout levels, volume for confirmation
# Discrete sizing 0.25 balances return and fee drag. Target: 50-150 total trades over 4 years = 12-37/year.

name = "6h_WilliamsAlligator_1d_WeeklyPivot_Volume_ATRStop_v1"
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
    
    # Calculate 1d Williams Alligator (Jaw=13, Teeth=8, Lips=5) SMMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for Alligator calculation
        return np.zeros(n)
    
    median_1d = (df_1d['high'].values + df_1d['low'].values) / 2
    
    # SMMA (Smoothed Moving Average) calculation
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_1d, 13)  # Jaw (Blue) - 13-period SMMA
    teeth = smma(median_1d, 8)  # Teeth (Red) - 8-period SMMA
    lips = smma(median_1d, 5)   # Lips (Green) - 5-period SMMA
    
    # Align Alligator lines to 6h timeframe (completed 1d bar only)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate 1d weekly pivot points (using prior week's OHLC)
    # For daily data, we approximate weekly by using 5-day aggregates
    high_5d = pd.Series(df_1d['high'].values).rolling(window=5, min_periods=5).max().shift(1).values
    low_5d = pd.Series(df_1d['low'].values).rolling(window=5, min_periods=5).min().shift(1).values
    close_5d = pd.Series(df_1d['close'].values).rolling(window=5, min_periods=5).mean().shift(1).values
    
    # Weekly pivot: (High + Low + Close) / 3
    weekly_pivot = (high_5d + low_5d + close_5d) / 3
    weekly_range = high_5d - low_5d
    
    # Weekly R3 and S3 levels
    r3_weekly = weekly_pivot + (weekly_range * 1.1)
    s3_weekly = weekly_pivot - (weekly_range * 1.1)
    
    # Align weekly pivot levels to 6h timeframe
    r3_weekly_aligned = align_htf_to_ltf(prices, df_1d, r3_weekly)
    s3_weekly_aligned = align_htf_to_ltf(prices, df_1d, s3_weekly)
    
    # 6h ATR(21) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup (need enough for Alligator and ATR calculations)
    start_idx = 50 + 5  # 50 for Alligator, 5 for ATR
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(r3_weekly_aligned[i]) or np.isnan(s3_weekly_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Alligator bullish: Lips > Teeth > Jaw
            alligator_bullish = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
            # Alligator bearish: Lips < Teeth < Jaw
            alligator_bearish = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
            
            # Long entry: price breaks above weekly R3 with Alligator bullish AND volume spike
            if close[i] > r3_weekly_aligned[i] and alligator_bullish and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                entry_bar = i
                highest_since_entry = high[i]
            # Short entry: price breaks below weekly S3 with Alligator bearish AND volume spike
            elif close[i] < s3_weekly_aligned[i] and alligator_bearish and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                entry_bar = i
                lowest_since_entry = low[i]
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, high[i])
            # ATR trailing stop: exit when price < highest_high_since_entry - 3.0 * ATR
            if close[i] < highest_since_entry - 3.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, low[i])
            # ATR trailing stop: exit when price > lowest_low_since_entry + 3.0 * ATR
            if close[i] > lowest_since_entry + 3.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals