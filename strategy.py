#!/usr/bin/env python3
"""
#100874 - 1h_Camarilla_R1_S1_Breakout_4hTrend_Volume
Hypothesis: 1-hour breakouts at daily R1/S1 levels with 4-hour EMA trend filter and volume confirmation.
Uses 4h for trend direction (EMA50), 1d for Camarilla pivot levels, and 1h for entry timing.
Targets 15-37 trades/year (60-150 total) to avoid fee drag. Works in bull (breakouts with trend) and bear (mean reversion to pivot).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla levels from previous day (to avoid look-ahead)
    daily_pivot = (high_1d + low_1d + close_1d) / 3
    daily_range = high_1d - low_1d
    daily_r1 = close_1d + daily_range * 1.1 / 12
    daily_s1 = close_1d - daily_range * 1.1 / 12
    
    # Align to 1h timeframe (previous day's levels for current period)
    camarilla_r1 = align_htf_to_ltf(prices, df_1d, daily_r1)
    camarilla_s1 = align_htf_to_ltf(prices, df_1d, daily_s1)
    camarilla_pivot = align_htf_to_ltf(prices, df_1d, daily_pivot)
    
    # Volume filter: volume > 1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(camarilla_r1[i]) or 
            np.isnan(camarilla_s1[i]) or np.isnan(camarilla_pivot[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above R1, above 4h EMA50, volume spike
        if (close[i] > camarilla_r1[i] and 
            close[i] > ema50_4h_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.20
            position = 1
        # Short condition: price breaks below S1, below 4h EMA50, volume spike
        elif (close[i] < camarilla_s1[i] and 
              close[i] < ema50_4h_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.20
            position = -1
        # Exit conditions: price returns to Camarilla Pivot (mean reversion)
        elif position == 1 and close[i] < camarilla_pivot[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > camarilla_pivot[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0