#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with 1w Camarilla pivot levels (R1/S1) as dynamic support/resistance.
Long when price breaks above 1w Camarilla R1 with volume confirmation and price > 12h EMA50 (trend filter).
Short when price breaks below 1w Camarilla S1 with volume confirmation and price < 12h EMA50.
Uses weekly Camarilla levels for institutional reference points, EMA50 for trend alignment,
and volume to confirm breakout strength. Designed to work in both bull (breakouts above R1 in uptrend)
and bear (breakdowns below S1 in downtrend) markets by requiring trend alignment.
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
    
    # Calculate 1w Camarilla pivot levels (R1, S1)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    r1_1w = close_1w + range_1w * 1.1 / 12.0
    s1_1w = close_1w - range_1w * 1.1 / 12.0
    
    # Calculate 12h EMA50 for trend filter
    close_s = pd.Series(close)
    ema50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 12h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1w Camarilla levels to 12h timeframe
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or 
            np.isnan(ema50[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * vol_ma_20[i]
        
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
            # Exit long: price falls back below 1w Camarilla S1 (opposite side)
            if close[i] < s1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 1w Camarilla R1 (opposite side)
            if close[i] > r1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1wCamarilla_R1S1_Breakout_Volume_EMA50_Trend"
timeframe = "12h"
leverage = 1.0