#!/usr/bin/env python3
# 6h_camellia_pivot_1d_trend_volume_v1
# Hypothesis: Fade at daily Camarilla R3/S3 levels, breakout continuation at R4/S4 with volume.
# Only trade when 1w trend aligns (price > weekly EMA50 for longs, < for shorts).
# Uses mean reversion at extremes and breakout continuation to work in both bull/bear.
# Target: 15-25 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camellia_pivot_1d_trend_volume_v1"
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
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (based on previous day)
    # Formula: Range = High - Low
    # R4 = Close + Range * 1.1/2
    # R3 = Close + Range * 1.1/4
    # R2 = Close + Range * 1.1/6
    # R1 = Close + Range * 1.1/12
    # PP = (High + Low + Close)/3
    # S1 = Close - Range * 1.1/12
    # S2 = Close - Range * 1.1/6
    # S3 = Close - Range * 1.1/4
    # S4 = Close - Range * 1.1/2
    range_1d = high_1d - low_1d
    
    r4 = close_1d + range_1d * 1.1 / 2
    r3 = close_1d + range_1d * 1.1 / 4
    s3 = close_1d - range_1d * 1.1 / 4
    s4 = close_1d - range_1d * 1.1 / 2
    
    # Align to 6h timeframe (use previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Get weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below S3 (mean reversion) OR weekly trend turns against us
            if (close[i] < s3_aligned[i]) or (close[i] < ema_50_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above R3 (mean reversion) OR weekly trend turns against us
            if (close[i] > r3_aligned[i]) or (close[i] > ema_50_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Mean reversion entry: fade at R3/S3 with volume
            # Short at R3: price crosses below R3 from above with volume confirmation in downtrend
            if (close[i] < r3_aligned[i]) and (close[i-1] >= r3_aligned[i-1]) and \
               (volume[i] > vol_ma[i]) and (close[i] < ema_50_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
            # Long at S3: price crosses above S3 from below with volume confirmation in uptrend
            elif (close[i] > s3_aligned[i]) and (close[i-1] <= s3_aligned[i-1]) and \
                 (volume[i] > vol_ma[i]) and (close[i] > ema_50_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Breakout continuation: break R4/S4 with volume in trend direction
            elif (close[i] > r4_aligned[i]) and (close[i-1] <= r4_aligned[i-1]) and \
                 (volume[i] > vol_ma[i]) and (close[i] > ema_50_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            elif (close[i] < s4_aligned[i]) and (close[i-1] >= s4_aligned[i-1]) and \
                 (volume[i] > vol_ma[i]) and (close[i] < ema_50_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals