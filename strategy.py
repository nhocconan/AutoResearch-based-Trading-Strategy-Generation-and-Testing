#!/usr/bin/env python3
# 12h_1d_camarilla_pivot_volume_v1
# Strategy: 12h Camarilla pivot levels from 1d with volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels (S3/S4 for longs, R3/R4 for shorts) act as strong support/resistance.
# Enter long when price touches S3/S4 with volume confirmation in uptrend (price > 1d EMA200).
# Enter short when price touches R3/R4 with volume confirmation in downtrend (price < 1d EMA200).
# Uses 12h close for signal generation to avoid look-ahead. Low frequency (~20-40/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_pivot_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d OHLC for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    typical = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    # Camarilla levels: S1=close - 1.1*range/12, S2=close - 1.1*range/6, S3=close - 1.1*range/4, S4=close - 1.1*range/2
    # R1=close + 1.1*range/12, R2=close + 1.1*range/6, R3=close + 1.1*range/4, R4=close + 1.1*range/2
    s3 = close_1d - 1.1 * range_hl / 4
    s4 = close_1d - 1.1 * range_hl / 2
    r3 = close_1d + 1.1 * range_hl / 4
    r4 = close_1d + 1.1 * range_hl / 2
    
    # Align Camarilla levels to 12h timeframe (wait for 1d bar to close)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    
    # 1d EMA(200) for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(ema_200_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 1d EMA200
        uptrend = close[i] > ema_200_1d_aligned[i]
        downtrend = close[i] < ema_200_1d_aligned[i]
        
        # Entry logic: Camarilla touch + volume + trend alignment
        # Long when price touches S3 or S4 in uptrend with volume
        long_condition = ((close[i] <= s3_aligned[i] * 1.001) or (close[i] <= s4_aligned[i] * 1.001)) and vol_confirm[i] and uptrend and position != 1
        # Short when price touches R3 or R4 in downtrend with volume
        short_condition = ((close[i] >= r3_aligned[i] * 0.999) or (close[i] >= r4_aligned[i] * 0.999)) and vol_confirm[i] and downtrend and position != -1
        
        if long_condition:
            position = 1
            signals[i] = 0.25
        elif short_condition:
            position = -1
            signals[i] = -0.25
        # Exit: price moves back toward midpoint (mean reversion) or trend change
        elif position == 1 and (close[i] >= (s3_aligned[i] + s4_aligned[i]) / 2 * 1.001 or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] <= (r3_aligned[i] + r4_aligned[i]) / 2 * 0.999 or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals