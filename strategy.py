#!/usr/bin/env python3
name = "6h_WeeklyPivot_Momentum_Confirmation"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot calculation
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Weekly resistance/support levels (standard pivot)
    r1_1w = pivot_1w + (range_1w * 1.0)
    s1_1w = pivot_1w - (range_1w * 1.0)
    r2_1w = pivot_1w + (range_1w * 2.0)
    s2_1w = pivot_1w - (range_1w * 2.0)
    
    # Align weekly levels to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Momentum confirmation: 6-period RSI on 6h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[0] = np.mean(gain[:14]) if len(gain) >= 14 else np.nan
    avg_loss[0] = np.mean(loss[:14]) if len(loss) >= 14 else np.nan
    
    for i in range(1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_ma20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        volume_ma20[i] = np.mean(volume[i-20:i])
    volume_filter = volume > 1.5 * volume_ma20
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(100, 20)  # Ensure sufficient data for calculations
    
    for i in range(start_idx, n):
        # Skip if any required values are NaN
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(r2_1w_aligned[i]) or 
            np.isnan(s2_1w_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price above weekly R1, RSI > 50 (bullish momentum), volume confirmation
            if close[i] > r1_1w_aligned[i] and rsi[i] > 50 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below weekly S1, RSI < 50 (bearish momentum), volume confirmation
            elif close[i] < s1_1w_aligned[i] and rsi[i] < 50 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price below weekly pivot or RSI < 40
            if close[i] < pivot_1w_aligned[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price above weekly pivot or RSI > 60
            if close[i] > pivot_1w_aligned[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals