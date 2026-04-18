#!/usr/bin/env python3
"""
12h_Pivot_R1S1_Breakout_Volume_v4
Strategy: 12h Camarilla pivot (R1/S1) breakout with volume confirmation and volatility filter.
Long: Break above R1 with volume > 1.5x average and volatility in normal range.
Short: Break below S1 with volume > 1.5x average and volatility in normal range.
Uses 1d Camarilla levels for structure, avoids overtrading via volatility filter.
Target: 15-25 trades/year per symbol (60-100 total over 4 years).
Works in bull/bear via volatility regime filter.
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
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]  # first day uses same day
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    # Camarilla levels: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    range_1d = prev_high - prev_low
    r1 = prev_close + range_1d * 1.1 / 12
    s1 = prev_close - range_1d * 1.1 / 12
    
    # Volatility filter: use ATR(20) to avoid choppy markets
    tr1 = np.maximum(high_1d - low_1d, np.absolute(high_1d - np.roll(close_1d, 1)))
    tr2 = np.absolute(np.roll(close_1d, 1) - low_1d)
    tr = np.maximum(tr1, tr2)
    tr[0] = high_1d[0] - low_1d[0]  # first day
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Align all daily data to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    atr_20_aligned = align_htf_to_ltf(prices, df_1d, atr_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(atr_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > 1.5 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
        
        # Volatility filter: avoid extreme volatility (stop hunting)
        vol_filter = atr_20_aligned[i] < pd.Series(atr_20_aligned).rolling(window=50, min_periods=50).mean().values[i] * 2
        
        if position == 0:
            # Long: price breaks above R1 with volume and volatility filter
            if close[i] > r1_aligned[i] and vol_confirm and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and volatility filter
            elif close[i] < s1_aligned[i] and vol_confirm and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below R1 or volatility spike
            if close[i] < r1_aligned[i] or not vol_filter:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above S1 or volatility spike
            if close[i] > s1_aligned[i] or not vol_filter:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Pivot_R1S1_Breakout_Volume_v4"
timeframe = "12h"
leverage = 1.0