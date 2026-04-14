#2025-05-26: Strategy for 6h timeframe using 1d timeframe for pivot levels and RSI
#Hypothesis: In ranging markets, price tends to revert to the daily pivot point. In trending markets,
#breaks above R1 or below S1 with momentum (RSI > 55 for long, < 45 for short) and volume confirmation
#can capture the start of a move. This combines mean reversion and momentum to work in both bull and bear markets.
#Uses daily pivot, RSI, and volume ratio to filter entries. Exits when price returns to pivot or momentum fades.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data (HTF) ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily pivot points (classic)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    r3 = high_1d + 2 * (pivot - low_1d)
    s3 = low_1d - 2 * (high_1d - pivot)
    
    # Align to 6h timeframe (waits for daily close)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 14-day RSI for momentum (daily)
    if len(close_1d) < 14:
        return np.zeros(n)
    
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    
    # Wilder's smoothing
    if len(close_1d) >= 14:
        avg_gain[13] = np.mean(gain[1:14])
        avg_loss[13] = np.mean(loss[1:14])
        for i in range(14, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.full_like(close_1d, np.nan)
    rsi14 = np.full_like(close_1d, np.nan)
    for i in range(13, len(close_1d)):
        if avg_loss[i] > 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi14[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi14[i] = 100 if avg_gain[i] > 0 else 0
    
    rsi14_aligned = align_htf_to_ltf(prices, df_1d, rsi14)
    
    # Precompute 20-period volume moving average for 6h
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(rsi14_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or 
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        volume_ratio = volume[i] / vol_ma_20[i]
        
        if position == 0:
            # Long: Price above S1 + RSI > 55 + volume surge
            if (close[i] > s1_aligned[i] and
                rsi14_aligned[i] > 55 and
                volume_ratio > 2.0):
                position = 1
                signals[i] = position_size
            # Short: Price below R1 + RSI < 45 + volume surge
            elif (close[i] < r1_aligned[i] and
                  rsi14_aligned[i] < 45 and
                  volume_ratio > 2.0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price returns to pivot (mean reversion) OR RSI < 40 (momentum fade)
            if (close[i] < pivot_aligned[i] or 
                rsi14_aligned[i] < 40):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price returns to pivot OR RSI > 60
            if (close[i] > pivot_aligned[i] or 
                rsi14_aligned[i] > 60):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_Pivot_R1S1_RSI14_Volume"
timeframe = "6h"
leverage = 1.0