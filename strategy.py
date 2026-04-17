#!/usr/bin/env python3
"""
1d_Weekly_Pivot_R1_S1_Breakout_Volume_Trend
Hypothesis: Weekly pivot levels (R1/S1) act as strong support/resistance on daily chart.
Long when price breaks above R1 with volume > 1.5x average and weekly close > weekly EMA34 (uptrend).
Short when price breaks below S1 with volume > 1.5x average and weekly close < weekly EMA34 (downtrend).
Exit on opposite signal. Position size: ±0.25. Uses 1d primary with 1w trend filter.
Designed for low-frequency, high-conviction trades in both bull and bear markets.
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
    
    # Calculate weekly pivot points (R1, S1, PP)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point calculation: PP = (H + L + C) / 3
    pp = (high_1w + low_1w + close_1w) / 3.0
    # R1 = 2*PP - L, S1 = 2*PP - H
    r1 = 2 * pp - low_1w
    s1 = 2 * pp - high_1w
    
    # Align weekly levels to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Weekly EMA34 for trend filter
    close_series_1w = pd.Series(close_1w)
    ema34_1w = close_series_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume confirmation (20-period MA on 1d)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(20, 34)  # Volume MA20, EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(volume_ma20[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Breakout conditions
        breakout_long = close[i] > r1_aligned[i]
        breakout_short = close[i] < s1_aligned[i]
        
        if position == 0:
            # Long: break above R1 + volume filter + weekly uptrend
            if breakout_long and volume_filter and close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 + volume filter + weekly downtrend
            elif breakout_short and volume_filter and close[i] < ema34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S1 (opposite level)
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R1 (opposite level)
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Pivot_R1_S1_Breakout_Volume_Trend"
timeframe = "1d"
leverage = 1.0