#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with weekly Camarilla pivot breakout + volume confirmation + 1d EMA50 trend filter.
Long when price breaks above weekly Camarilla R4 with volume confirmation and price > 1d EMA50 (uptrend).
Short when price breaks below weekly Camarilla S4 with volume confirmation and price < 1d EMA50 (downtrend).
Exit when price returns to the weekly Camarilla midpoint (R3/S3 average).
Uses weekly timeframe for structure (reduces noise) and 6h for entry timing and trend filter.
Designed to capture strong institutional breakouts with volume confirmation while avoiding false breakouts.
Weekly pivots are more significant than daily pivots and work well in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivot calculation (need weekly high/low/close)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla pivot levels
    # Pivot = (High + Low + Close) / 3
    # Range = High - Low
    # R4 = Pivot + Range * 1.1/2
    # S4 = Pivot - Range * 1.1/2
    # R3 = Pivot + Range * 1.1/4
    # S3 = Pivot - Range * 1.1/4
    # Midpoint (for exit) = (R3 + S3) / 2 = Pivot
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    r4_1w = pivot_1w + range_1w * 1.1 / 2.0
    s4_1w = pivot_1w - range_1w * 1.1 / 2.0
    r3_1w = pivot_1w + range_1w * 1.1 / 4.0
    s3_1w = pivot_1w - range_1w * 1.1 / 4.0
    midpoint_1w = pivot_1w  # (R3 + S3) / 2 simplifies to pivot
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 6h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly Camarilla levels to 6h timeframe
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    midpoint_1w_aligned = align_htf_to_ltf(prices, df_1w, midpoint_1w)
    
    # Align 1d EMA50 to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r4_1w_aligned[i]) or 
            np.isnan(s4_1w_aligned[i]) or 
            np.isnan(midpoint_1w_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above weekly Camarilla R4 with volume and uptrend (price > EMA50)
            if (close[i] > r4_1w_aligned[i] and 
                volume_confirmed and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Camarilla S4 with volume and downtrend (price < EMA50)
            elif (close[i] < s4_1w_aligned[i] and 
                  volume_confirmed and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to or below weekly Camarilla midpoint (pivot)
            if close[i] <= midpoint_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to or above weekly Camarilla midpoint (pivot)
            if close[i] >= midpoint_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1wCamarilla_R4S4_Breakout_Volume_1dEMA50_Trend"
timeframe = "6h"
leverage = 1.0