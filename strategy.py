#!/usr/bin/env python3
"""
4h_Pivot_R1S1_R2S2_Breakout_Volume_Trend
Hypothesis: Price breaks above/below Camarilla pivot levels (R1/S1/R2/S2) with volume confirmation and 4h EMA trend filter.
Camarilla levels derived from daily OHLC, providing institutional support/resistance. Works in both bull and bear markets by
buying strength above resistance in uptrends and selling weakness below support in downtrends. Volume filter ensures breakout
conviction. Target: 20-30 trades/year (80-120 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily OHLC for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formula: range = high - low
    # R1 = close + (range * 1.1/12)
    # S1 = close - (range * 1.1/12)
    # R2 = close + (range * 1.1/6)
    # S2 = close - (range * 1.1/6)
    range_1d = high_1d - low_1d
    r1 = close_1d + (range_1d * 1.1 / 12)
    s1 = close_1d - (range_1d * 1.1 / 12)
    r2 = close_1d + (range_1d * 1.1 / 6)
    s2 = close_1d - (range_1d * 1.1 / 6)
    
    # Align daily levels to 4h timeframe (wait for daily close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # EMA34 trend filter on 4h
    ema_34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume filter: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 34  # Warmup for EMA34
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(ema_34[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        ema_val = ema_34[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: price breaks above R1 or R2 with volume in uptrend
            if (price > r1_val or price > r2_val) and vol_ok and price > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 or S2 with volume in downtrend
            elif (price < s1_val or price < s2_val) and vol_ok and price < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price returns below S1 or trend reverses
            if price < s1_val or price < ema_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price returns above R1 or trend reverses
            if price > r1_val or price > ema_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Pivot_R1S1_R2S2_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0