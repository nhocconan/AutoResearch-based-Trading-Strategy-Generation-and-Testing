#!/usr/bin/env python3
"""
Strategy: 6h_Adaptive_Camarilla_Pivot_Trend_Filter
Timeframe: 6h
Hypothesis: 
- Uses daily Camarilla pivot levels (R3/S3, R4/S4) for reversal and breakout signals
- Long when price breaks above R4 with volume confirmation and 12h EMA50 uptrend
- Short when price breaks below S4 with volume confirmation and 12h EMA50 downtrend
- Reversals at R3/S3 when price rejects level with volume divergence
- Designed to work in trending markets (breakouts) and ranging markets (reversals)
- Filters out low-conviction moves using volume and trend alignment
"""

name = "6h_Adaptive_Camarilla_Pivot_Trend_Filter"
timeframe = "6h"
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
    
    # 12h EMA50 for trend filter
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate daily pivot point: (H + L + C) / 3
    pivot = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Calculate Camarilla levels
    range_ = df_1d['high'] - df_1d['low']
    R3 = pivot + (range_ * 1.1 / 2)
    S3 = pivot - (range_ * 1.1 / 2)
    R4 = pivot + (range_ * 1.1)
    S4 = pivot - (range_ * 1.1)
    
    # Align daily levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot.values)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3.values)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3.values)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4.values)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4.values)
    
    # Volume confirmation: current volume > 1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean()
    vol_spike = volume > (1.8 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Breakout long: price breaks above R4 with volume spike and uptrend
            if (close[i] > R4_aligned[i] and 
                vol_spike[i] and 
                close[i] > ema50[i]):
                signals[i] = 0.25
                position = 1
            # Breakout short: price breaks below S4 with volume spike and downtrend
            elif (close[i] < S4_aligned[i] and 
                  vol_spike[i] and 
                  close[i] < ema50[i]):
                signals[i] = -0.25
                position = -1
            # Reversal long: price rejects S3 with volume divergence (mean reversion)
            elif (close[i] < S3_aligned[i] and 
                  close[i-1] <= S3_aligned[i-1] and  # Was at or below S3
                  close[i] > close[i-1] and          # Now reversing up
                  vol_spike[i] and                 # Volume on reversal
                  close[i] < ema50[i]):            # Still in downtrend (fade the trend)
                signals[i] = 0.20
                position = 1
            # Reversal short: price rejects R3 with volume divergence
            elif (close[i] > R3_aligned[i] and 
                  close[i-1] >= R3_aligned[i-1] and  # Was at or above R3
                  close[i] < close[i-1] and          # Now reversing down
                  vol_spike[i] and                 # Volume on reversal
                  close[i] > ema50[i]):            # Still in uptrend (fade the trend)
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S3 (failure) OR trend turns bearish
            if (close[i] < S3_aligned[i]) or (close[i] < ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if close[i] > R4_aligned[i] else 0.20
        
        elif position == -1:
            # Exit short: price breaks above R3 (failure) OR trend turns bullish
            if (close[i] > R3_aligned[i]) or (close[i] > ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25 if close[i] < S4_aligned[i] else -0.20
    
    return signals