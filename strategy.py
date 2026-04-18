#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R (14) with daily pivot point (PP) as dynamic support/resistance.
# Williams %R identifies overbought/oversold conditions; pivot points provide institutional levels.
# Long when %R < -80 (oversold) and price > PP; Short when %R > -20 (overbought) and price < PP.
# Uses volume confirmation and session filter to reduce false signals.
# Designed for low trade frequency (15-30/year) to minimize fee drag in 12h timeframe.
# Works in both bull and bear markets by fading extremes at key institutional levels.
name = "12h_WilliamsR_PivotPoint_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot point calculation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot point: PP = (H + L + C) / 3
    # Use previous day's OHLC to avoid look-ahead
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    pivot_point = (prev_high + prev_low + prev_close) / 3
    pivot_point_aligned = align_htf_to_ltf(prices, df_1d, pivot_point)
    
    # Calculate Williams %R (14) on 12h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Williams %R and volume MA calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_point_aligned[i]) or
            np.isnan(williams_r[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) and price above pivot point
            if vol_confirm and williams_r[i] < -80 and close[i] > pivot_point_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) and price below pivot point
            elif vol_confirm and williams_r[i] > -20 and close[i] < pivot_point_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R rises above -50 (momentum fading) or price below pivot
            if williams_r[i] > -50 or close[i] < pivot_point_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R falls below -50 (momentum fading) or price above pivot
            if williams_r[i] < -50 or close[i] > pivot_point_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals