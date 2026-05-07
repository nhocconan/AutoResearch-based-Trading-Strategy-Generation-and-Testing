#!/usr/bin/env python3
# 6h_Camarilla_R4_S4_Breakout_12hTrend_Volume
# Hypothesis: Camarilla pivot levels from daily timeframe identify key support/resistance.
# Breakouts at R4/S4 levels (extreme levels) with 12h trend alignment and volume confirmation
# capture strong momentum moves. Works in bull/bear markets by following 12h trend direction.
# Target: 15-35 trades/year per symbol with strict entry conditions to minimize fee drag.

name = "6h_Camarilla_R4_S4_Breakout_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) == 0:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Using previous day's data to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and ranges
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r4_1d = close_1d + (range_1d * 1.1 / 2.0)
    r3_1d = close_1d + (range_1d * 1.1 / 4.0)
    s3_1d = close_1d - (range_1d * 1.1 / 4.0)
    s4_1d = close_1d - (range_1d * 1.1 / 2.0)
    
    # Align Camarilla levels to 6h timeframe (using previous day's levels)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume spike detection: 1.8x average volume (60-period = ~15 days on 6h chart)
    vol_ma = pd.Series(volume).rolling(window=60, min_periods=60).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 60)  # Ensure we have EMA50 and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R4, price above 12h EMA50 (uptrend), volume spike
            if (high[i] > r4_1d_aligned[i-1] and 
                close[i] > ema50_12h_aligned[i] and 
                volume[i] > 1.8 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4, price below 12h EMA50 (downtrend), volume spike
            elif (low[i] < s4_1d_aligned[i-1] and 
                  close[i] < ema50_12h_aligned[i] and 
                  volume[i] > 1.8 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below R3 OR price crosses below 12h EMA50
            if (low[i] < r3_1d_aligned[i-1] or close[i] < ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above S3 OR price crosses above 12h EMA50
            if (high[i] > s3_1d_aligned[i-1] or close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals