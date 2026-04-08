#!/usr/bin/env python3
# 6h_12h_1d_pivot_volume_v1
# Hypothesis: Use 12h Camarilla pivot levels for entry/exit signals with 1d trend filter and volume confirmation.
# Camarilla levels provide high-probability reversal points; 1d trend filters for direction; volume confirms institutional interest.
# Works in bull markets (trend continuation from S3/R3) and bear markets (mean reversion at S4/R4 levels).
# Target: 15-30 trades/year per symbol (60-120 total over 4 years) by requiring multi-timeframe alignment and volume filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for price action
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Get 12h data for Camarilla pivot levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 12h Camarilla pivot levels (using previous 12h bar's high, low, close)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Pivot point and Camarilla levels
    pivot = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    
    # Camarilla levels (S1-S4, R1-R4)
    r4 = close_12h + (range_12h * 1.500)
    r3 = close_12h + (range_12h * 1.250)
    r2 = close_12h + (range_12h * 1.166)
    r1 = close_12h + (range_12h * 1.083)
    s1 = close_12h - (range_12h * 1.083)
    s2 = close_12h - (range_12h * 1.166)
    s3 = close_12h - (range_12h * 1.250)
    s4 = close_12h - (range_12h * 1.500)
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    
    # 1d trend filter: EMA(20) vs EMA(50)
    ema_20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.5x average of last 24 periods (1 day in 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or \
           np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or \
           np.isnan(ema_20_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below S3 or breaks below S4 (stop loss)
            if close[i] < s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above R3 or breaks above R4 (stop loss)
            if close[i] > r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price touches S3 with 1d uptrend bias and volume
            if (close[i] <= s3_aligned[i] * 1.002 and  # Within 0.2% of S3
                ema_20_1d_aligned[i] > ema_50_1d_aligned[i] and  # 1d uptrend
                vol_confirm[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price touches R3 with 1d downtrend bias and volume
            elif (close[i] >= r3_aligned[i] * 0.998 and  # Within 0.2% of R3
                  ema_20_1d_aligned[i] < ema_50_1d_aligned[i] and  # 1d downtrend
                  vol_confirm[i]):
                position = -1
                signals[i] = -0.25
    
    return signals