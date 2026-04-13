#!/usr/bin/env python3
"""
1h_4d_1d_Camarilla_Pivot_Breakout_Volume
Hypothesis: Use 1-day and 4-hour charts to determine trend direction via Camarilla pivot breakouts, 
then use 1-hour chart for precise entry timing with volume confirmation. 
This reduces false breakouts by requiring alignment between 1d and 4h trends.
Target: 15-30 trades/year per symbol (60-120 over 4 years).
Works in bull markets via upward breakouts and bear markets via downward breakdowns.
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
    
    # Get daily data for trend direction
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 4h data for intermediate trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from daily chart (trend filter)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    close_prev_1d = np.roll(close_1d, 1)
    close_prev_1d[0] = close_1d[0]
    
    range_1d = high_1d - low_1d
    
    # Daily Camarilla levels
    R4_1d = close_prev_1d + (range_1d * 1.5000 / 2)
    S4_1d = close_prev_1d - (range_1d * 1.5000 / 2)
    
    # Align daily levels to 1h
    R4_1d_aligned = align_htf_to_ltf(prices, df_1d, R4_1d)
    S4_1d_aligned = align_htf_to_ltf(prices, df_1d, S4_1d)
    
    # Calculate Camarilla levels from 4h chart (intermediate filter)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    close_prev_4h = np.roll(close_4h, 1)
    close_prev_4h[0] = close_4h[0]
    
    range_4h = high_4h - low_4h
    
    # 4h Camarilla levels
    R4_4h = close_prev_4h + (range_4h * 1.5000 / 2)
    S4_4h = close_prev_4h - (range_4h * 1.5000 / 2)
    
    # Align 4h levels to 1h
    R4_4h_aligned = align_htf_to_ltf(prices, df_4h, R4_4h)
    S4_4h_aligned = align_htf_to_ltf(prices, df_4h, S4_4h)
    
    # Volume confirmation: current volume > 2.0x 24-period average
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean()
    volume_expansion = volume > (vol_ma_24 * 2.0)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.20
    
    for i in range(24, n):
        # Skip if any required data is not ready or outside session
        if (np.isnan(R4_1d_aligned[i]) or np.isnan(S4_1d_aligned[i]) or
            np.isnan(R4_4h_aligned[i]) or np.isnan(S4_4h_aligned[i]) or
            np.isnan(volume_expansion[i]) or not session_filter[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from daily chart
        daily_bullish = close[i] > R4_1d_aligned[i]
        daily_bearish = close[i] < S4_1d_aligned[i]
        
        # Require 4h chart to agree with daily trend
        fourh_bullish = close[i] > R4_4h_aligned[i]
        fourh_bearish = close[i] < S4_4h_aligned[i]
        
        # Long setup: daily and 4h bullish + 1h breakout above R4 with volume
        long_setup = daily_bullish and fourh_bullish and close[i] > R4_4h_aligned[i] and volume_expansion[i]
        
        # Short setup: daily and 4h bearish + 1h breakdown below S4 with volume
        short_setup = daily_bearish and fourh_bearish and close[i] < S4_4h_aligned[i] and volume_expansion[i]
        
        if long_setup and position != 1:
            position = 1
            signals[i] = position_size
        elif short_setup and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "1h_4d_1d_Camarilla_Pivot_Breakout_Volume"
timeframe = "1h"
leverage = 1.0