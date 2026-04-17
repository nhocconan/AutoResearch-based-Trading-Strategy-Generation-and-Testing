#!/usr/bin/env python3
"""
4h_Pivot_R1_S1_Breakout_Volume_Trend_HTF_v3
Hypothesis: Camarilla pivot levels from 1d provide high-probability breakout zones. 
Long when price breaks above R1 with volume confirmation and 1d close > EMA50 (uptrend).
Short when price breaks below S1 with volume confirmation and 1d close < EMA50 (downtrend).
Exit on opposite signal or close back inside the pivot range (between S1 and R1).
Position size: ±0.25. Uses 4h primary with 1d trend filter.
Designed to work in both bull (breakouts in uptrend) and bear (breakdowns in downtrend).
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
    
    # Volume confirmation (10-period MA on 4h)
    volume_ma10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    # Get 1d data for pivot levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # R4 = close + (high - low) * 1.5000
    # R3 = close + (high - low) * 1.2500
    # R2 = close + (high - low) * 1.1666
    # R1 = close + (high - low) * 1.0833
    # PP = (high + low + close) / 3
    # S1 = close - (high - low) * 1.0833
    # S2 = close - (high - low) * 1.1666
    # S3 = close - (high - low) * 1.2500
    # S4 = close - (high - low) * 1.5000
    pivot_range = high_1d - low_1d
    r1_1d = close_1d + pivot_range * 1.0833
    s1_1d = close_1d - pivot_range * 1.0833
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Calculate 1d EMA50 for trend filter
    close_series_1d = pd.Series(close_1d)
    ema50_1d = close_series_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(10, 50)  # volume MA10, EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma10[i]) or 
            np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 10-period average
        volume_filter = volume[i] > (1.5 * volume_ma10[i])
        
        if position == 0:
            # Long: price breaks above R1 + volume filter + 1d uptrend
            if close[i] > r1_1d_aligned[i] and volume_filter and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume filter + 1d downtrend
            elif close[i] < s1_1d_aligned[i] and volume_filter and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price closes back below R1 (or below PP for earlier exit)
            if close[i] < r1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price closes back above S1 (or above PP for earlier exit)
            if close[i] > s1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Pivot_R1_S1_Breakout_Volume_Trend_HTF_v3"
timeframe = "4h"
leverage = 1.0