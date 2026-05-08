#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Camarilla R1, S1, R3, S3 (based on previous day's range)
    # R1 = close + (high - low) * 1.12 / 12
    # S1 = close - (high - low) * 1.12 / 12
    # R3 = close + (high - low) * 1.12 / 4
    # S3 = close - (high - low) * 1.12 / 4
    range_1d = high_1d - low_1d
    r1 = close_1d + range_1d * 1.12 / 12
    s1 = close_1d - range_1d * 1.12 / 12
    r3 = close_1d + range_1d * 1.12 / 4
    s3 = close_1d - range_1d * 1.12 / 4
    
    # Align Camarilla levels to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    r3_4h = align_htf_to_ltf(prices, df_1d, r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate volume spike filter (20-period volume MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for EMA34
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or 
            np.isnan(ema34_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Price breaks above R1 with volume spike AND price > EMA34 (uptrend)
            long_cond = (close[i] > r1_4h[i]) and vol_spike[i] and (close[i] > ema34_4h[i])
            
            # Short entry: Price breaks below S1 with volume spike AND price < EMA34 (downtrend)
            short_cond = (close[i] < s1_4h[i]) and vol_spike[i] and (close[i] < ema34_4h[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price crosses below S1 (reversion to mean)
            if close[i] < s1_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price crosses above R1 (reversion to mean)
            if close[i] > r1_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R1/S1 levels act as intraday support/resistance with mean reversion properties.
# Long when price breaks above R1 with volume confirmation in an uptrend (price > EMA34).
# Short when price breaks below S1 with volume confirmation in a downtrend (price < EMA34).
# Exits when price reverts to opposite S1/R1 levels.
# Works in both bull and bear markets by following the daily trend filter.
# Volume spike ensures institutional participation and reduces false breakouts.
# Target: 20-50 trades per year to minimize fee decay while capturing meaningful moves.