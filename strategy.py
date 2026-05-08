#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla Pivot Points with volume confirmation and trend filter.
# Long when price breaks above R3 level with volume confirmation and 1d EMA50 uptrend.
# Short when price breaks below S3 level with volume confirmation and 1d EMA50 downtrend.
# Exit when price returns to Pivot level or opposite S1/R1 level.
# Designed for low trade frequency (20-40/year) to avoid fee drag. Works in trending markets via trend filter.

name = "4h_1dCamarilla_R3S3_Breakout_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla Pivots and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Camarilla Pivot Points (using previous day's OHLC)
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    # Camarilla levels
    R1 = close_1d + (range_hl * 1.1 / 12)
    R2 = close_1d + (range_hl * 1.1 / 6)
    R3 = close_1d + (range_hl * 1.1 / 4)
    R4 = close_1d + (range_hl * 1.1 / 2)
    
    S1 = close_1d - (range_hl * 1.1 / 12)
    S2 = close_1d - (range_hl * 1.1 / 6)
    S3 = close_1d - (range_hl * 1.1 / 4)
    S4 = close_1d - (range_hl * 1.1 / 2)
    
    # Calculate 1d EMA50 for trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    Pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: 4h volume > 1.3x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or 
            np.isnan(Pivot_aligned[i]) or
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R3 with volume confirmation and uptrend
            if (close[i] > R3_aligned[i] and 
                vol_confirm[i] and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 with volume confirmation and downtrend
            elif (close[i] < S3_aligned[i] and 
                  vol_confirm[i] and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Pivot or drops below S1
            if close[i] <= Pivot_aligned[i] or close[i] < S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Pivot or rises above R1
            if close[i] >= Pivot_aligned[i] or close[i] > R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals