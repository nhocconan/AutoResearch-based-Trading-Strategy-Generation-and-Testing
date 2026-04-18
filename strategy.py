#!/usr/bin/env python3
"""
4h Camarilla Pivot + Volume Spike + RSI Filter
Uses Camarilla pivot levels from 1d timeframe with volume confirmation and RSI filter.
Designed for low trade frequency with strong edge in both trending and ranging markets.
Targets 25-40 trades per year by requiring confluence of price at pivot levels, volume spike, and RSI momentum.
"""

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
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (R3, R2, R1, S1, S2, S3) from previous day
    # Using previous day's range to calculate today's levels
    range_1d = high_1d - low_1d
    # Shift by 1 to use previous day's data for today's levels
    range_1d_prev = np.roll(range_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    
    # First value will be invalid due to roll, handle with nans
    range_1d_prev[0] = np.nan
    close_1d_prev[0] = np.nan
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    
    # Camarilla levels calculation
    r3 = close_1d_prev + (range_1d_prev * 1.1 / 4)
    r2 = close_1d_prev + (range_1d_prev * 1.1 / 6)
    r1 = close_1d_prev + (range_1d_prev * 1.1 / 12)
    s1 = close_1d_prev - (range_1d_prev * 1.1 / 12)
    s2 = close_1d_prev - (range_1d_prev * 1.1 / 6)
    s3 = close_1d_prev - (range_1d_prev * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # RSI calculation (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike detection (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(r3_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price at S1 level with volume spike and RSI > 50 (bullish momentum)
            if (abs(price - s1_aligned[i]) < (s1_aligned[i] * 0.001) and  # Within 0.1% of S1
                volume_spike[i] and 
                rsi[i] > 50):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price at R1 level with volume spike and RSI < 50 (bearish momentum)
            elif (abs(price - r1_aligned[i]) < (r1_aligned[i] * 0.001) and  # Within 0.1% of R1
                  volume_spike[i] and 
                  rsi[i] < 50):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price reaches S2 (support) or RSI turns bearish
            if price <= s2_aligned[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price reaches R2 (resistance) or RSI turns bullish
            if price >= r2_aligned[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_Pivot_Volume_Spike_RSI_Filter"
timeframe = "4h"
leverage = 1.0