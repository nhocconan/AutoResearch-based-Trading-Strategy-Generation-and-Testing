#!/usr/bin/env python3
"""
1h_Camarilla_R3S3_Breakout_4hTrend_Volume
Hypothesis: 1h breakouts at 4h Camarilla R3/S3 levels with 4h EMA50 trend filter and volume spike.
Uses 4h for trend and structure, 1h for precise entry timing to reduce whipsaw.
Volume confirms institutional interest. Target: 60-150 trades over 4 years (15-37/year).
Works in bull (buy R3 breakouts in uptrend) and bear (sell S3 breakdowns in downtrend).
"""

name = "1h_Camarilla_R3S3_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Load 4h data ONCE for Camarilla levels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels: R3, S3 (outer levels for fewer, stronger signals)
    hl_range = high_4h - low_4h
    r3 = close_4h + hl_range * 1.5000
    s3 = close_4h - hl_range * 1.5000
    
    # Align Camarilla levels to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3)
    
    # Volume filter: 20-period EMA for spike detection (using 1h volume)
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 2.0  # Higher threshold for fewer trades
    
    # Session filter: 08-20 UTC (reduce noise, focus on active hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_ok = (hours >= 8) & (hours <= 20)
    
    # Fixed position size to minimize churn
    position_size = 0.20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_ok[i]) or np.isnan(session_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Require session and volume
        if not (session_ok[i] and volume_ok[i]):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_ema4h = close[i] > ema50_4h_aligned[i]
        price_below_ema4h = close[i] < ema50_4h_aligned[i]
        breakout_long = close[i] > r3_aligned[i]
        breakout_short = close[i] < s3_aligned[i]
        
        if position == 0:
            # Long: Price breaks above R3 + above 4h EMA50 + volume + session
            if breakout_long and price_above_ema4h:
                signals[i] = position_size
                position = 1
            # Short: Price breaks below S3 + below 4h EMA50 + volume + session
            elif breakout_short and price_below_ema4h:
                signals[i] = -position_size
                position = -1
        else:
            # Exit: Price crosses opposite Camarilla level OR trend reverses
            if position == 1:
                if close[i] < s3_aligned[i] or close[i] < ema50_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if close[i] > r3_aligned[i] or close[i] > ema50_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals