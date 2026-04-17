#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_Breakout_Volume_Confirmation
Hypothesis: Camarilla pivot levels (R1/S1) act as strong support/resistance on daily timeframe.
Breakouts above R1 or below S1 with volume confirmation yield sustained moves.
Long when price breaks above R1 + volume > 2x average, short when breaks below S1 + volume > 2x average.
Exit on opposite signal or when price returns to pivot point (PP).
Uses weekly trend filter: only take longs when weekly close > weekly EMA20, shorts when weekly close < weekly EMA20.
Designed to work in both bull (breakouts in uptrend) and bear (breakouts in downtrend) with low trade frequency.
Position size: ±0.25.
"""

import numpy as np
import pandas as pd
from math import ceil
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily Camarilla pivot levels
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1/12
    # S1 = C - (H - L) * 1.1/12
    # R2 = C + (H - L) * 1.1/6
    # S2 = C - (H - L) * 1.1/6
    # We'll use R1 and S1 for breakouts
    pp = (high + low + close) / 3.0
    r1 = close + (high - low) * 1.1 / 12.0
    s1 = close - (high - low) * 1.1 / 12.0
    
    # Volume confirmation: 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA20 for trend filter
    close_series_1w = pd.Series(close_1w)
    ema20_1w = close_series_1w.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly EMA to daily timeframe
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(20, 20)  # volume MA20, weekly EMA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1[i]) or 
            np.isnan(s1[i]) or 
            np.isnan(pp[i]) or 
            np.isnan(volume_ma20[i]) or 
            np.isnan(ema20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2x 20-period average
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        # Breakout conditions
        breakout_long = close[i] > r1[i] and volume_filter
        breakout_short = close[i] < s1[i] and volume_filter
        
        # Mean reversion exit: price returns to pivot point
        reversion_to_pp = abs(close[i] - pp[i]) < (pp[i] * 0.005)  # within 0.5% of PP
        
        if position == 0:
            # Long: breakout above R1 + volume filter + weekly uptrend
            if breakout_long and close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S1 + volume filter + weekly downtrend
            elif breakout_short and close[i] < ema20_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: breakout fails (return to PP) or opposite breakout
            if reversion_to_pp or breakout_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: breakout fails (return to PP) or opposite breakout
            if reversion_to_pp or breakout_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_Pivot_Breakout_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0