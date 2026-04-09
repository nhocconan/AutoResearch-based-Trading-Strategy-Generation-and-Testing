#!/usr/bin/env python3
# 6h_market_structure_bounce_v1
# Hypothesis: Combines 1-day market structure (higher highs/lows) with 6-hour pullbacks to EMA21 for trend continuation entries in both bull and bear markets.
# Uses 1d swing points for trend direction, 6h EMA21 pullback for entry, and volume confirmation.
# Target: 20-40 trades/year (80-160 over 4 years) with controlled risk.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_market_structure_bounce_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA21 on 6h for pullback entries
    alpha = 2 / (21 + 1)
    ema21 = np.zeros(n)
    ema21[0] = close[0]
    for i in range(1, n):
        ema21[i] = alpha * close[i] + (1 - alpha) * ema21[i-1]
    
    # Get daily data for market structure
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Identify swing highs and lows on daily
    swing_high = np.zeros(len(df_1d), dtype=bool)
    swing_low = np.zeros(len(df_1d), dtype=bool)
    
    for i in range(2, len(df_1d)-2):
        # Swing high: higher than 2 bars before and after
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            swing_high[i] = True
        # Swing low: lower than 2 bars before and after
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            swing_low[i] = True
    
    # Determine trend based on last two swing points
    trend = np.zeros(len(df_1d))  # 1=uptrend, -1=downtrend, 0=undefined
    last_swing_high_idx = -1
    last_swing_low_idx = -1
    
    for i in range(len(df_1d)):
        if swing_high[i]:
            last_swing_high_idx = i
        if swing_low[i]:
            last_swing_low_idx = i
        
        if last_swing_high_idx != -1 and last_swing_low_idx != -1:
            if last_swing_high_idx > last_swing_low_idx:
                trend[i] = 1  # Uptrend: last swing high is more recent
            else:
                trend[i] = -1  # Downtrend: last swing low is more recent
    
    # Align trend to 6h timeframe
    trend_6h = align_htf_to_ltf(prices, df_1d, trend)
    
    # Volume confirmation: 20-period average
    vol_ma_20 = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if np.isnan(ema21[i]) or np.isnan(trend_6h[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 1:  # Long position
            # Exit: trend changes to downtrend or price breaks below EMA21
            if trend_6h[i] == -1 or close[i] < ema21[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: trend changes to uptrend or price breaks above EMA21
            if trend_6h[i] == 1 or close[i] > ema21[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: uptrend + pullback to EMA21 + volume
            if (trend_6h[i] == 1 and 
                low[i] <= ema21[i] * 1.005 and  # Allow small tolerance
                close[i] > ema21[i] and
                vol_ok):
                position = 1
                signals[i] = 0.25
            # Enter short: downtrend + pullback to EMA21 + volume
            elif (trend_6h[i] == -1 and 
                  high[i] >= ema21[i] * 0.995 and  # Allow small tolerance
                  close[i] < ema21[i] and
                  vol_ok):
                position = -1
                signals[i] = -0.25
    
    return signals