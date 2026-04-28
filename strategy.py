#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike_v2
Hypothesis: Refined version with stricter entry conditions to reduce trade frequency (target 15-25 trades/year). Uses daily R3/S3 levels with 12h EMA50 trend filter and volume spike (>2.5x average). Adds minimum holding period of 8 bars to reduce whipsaw. Designed to work in both bull and bear markets by following the 12h trend direction while avoiding false breakouts.
"""

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
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla pivot levels from previous day
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_ = df_1d['high'] - df_1d['low']
    R3 = typical_price + (range_ * 1.1 / 2)
    S3 = typical_price - (range_ * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3.values)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3.values)
    
    # Volume confirmation: >2.5x 24-period MA (4 days of 4h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = 50  # Wait for EMA50 to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(R3_aligned[i]) or
            np.isnan(S3_aligned[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        # Trend filter: price above/below 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation (>2.5x average)
        vol_confirm = volume[i] > (2.5 * vol_ma_24[i])
        
        # Breakout conditions at R3/S3
        long_breakout = close[i] > R3_aligned[i] and vol_confirm and uptrend
        short_breakout = close[i] < S3_aligned[i] and vol_confirm and downtrend
        
        # Exit conditions: return to midpoint of R3/S3
        midpoint = (R3_aligned[i] + S3_aligned[i]) / 2
        long_exit = close[i] < midpoint
        short_exit = close[i] > midpoint
        
        # Minimum holding period: 8 bars (32 hours)
        min_hold = bars_since_entry >= 8
        
        if long_breakout and position <= 0 and min_hold:
            signals[i] = 0.25
            position = 1
            bars_since_entry = 0
        elif short_breakout and position >= 0 and min_hold:
            signals[i] = -0.25
            position = -1
            bars_since_entry = 0
        elif long_exit and position == 1 and min_hold:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        elif short_exit and position == -1 and min_hold:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        
        # Increment bars since entry if in a position
        if position != 0:
            bars_since_entry += 1
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0