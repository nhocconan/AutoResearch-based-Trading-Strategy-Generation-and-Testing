#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with 1w Camarilla R1/S1 breakout + volume confirmation + trend filter (price > 1d EMA50).
Long when price breaks above 1w Camarilla R1 level with volume confirmation and price > 1d EMA50 (uptrend).
Short when price breaks below 1w Camarilla S1 level with volume confirmation and price < 1d EMA50 (downtrend).
Exit when price returns to the 1w Camarilla midpoint (mean reversion to pivot center).
Uses 1w timeframe for structure (reduces noise) and 12h for entry timing and trend filter.
Designed to capture medium-term breakouts with institutional volume while avoiding false breakouts in choppy markets.
"""

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
    
    # Get 1w data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Camarilla pivot levels
    # Pivot = (high + low + close) / 3
    # Range = high - low
    # R1 = Pivot + (Range * 1.1 / 12)
    # S1 = Pivot - (Range * 1.1 / 12)
    # Mid = Pivot
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    r1_1w = pivot_1w + (range_1w * 1.1 / 12.0)
    s1_1w = pivot_1w - (range_1w * 1.1 / 12.0)
    mid_1w = pivot_1w
    
    # Calculate 1d EMA50 for trend filter
    close_s = pd.Series(close)
    ema50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 12h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1w Camarilla levels to 12h timeframe
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    mid_1w_aligned = align_htf_to_ltf(prices, df_1w, mid_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or 
            np.isnan(mid_1w_aligned[i]) or 
            np.isnan(ema50[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above 1w Camarilla R1 with volume and uptrend (price > EMA50)
            if (close[i] > r1_1w_aligned[i] and 
                volume_confirmed and 
                close[i] > ema50[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1w Camarilla S1 with volume and downtrend (price < EMA50)
            elif (close[i] < s1_1w_aligned[i] and 
                  volume_confirmed and 
                  close[i] < ema50[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to or below 1w Camarilla midpoint
            if close[i] <= mid_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to or above 1w Camarilla midpoint
            if close[i] >= mid_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1wCamarilla_R1S1_Breakout_Volume_1dEMA50_Trend"
timeframe = "12h"
leverage = 1.0