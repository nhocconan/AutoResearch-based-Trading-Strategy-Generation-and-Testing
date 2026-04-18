#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_Volume_Regime_v2
Hypothesis: Use 1d Camarilla pivot levels (S1/R1) with volume confirmation and chop regime filter on 12h timeframe.
Go long when price crosses above S1 with volume > 1.5x average and CHOP > 61.8 (ranging market).
Go short when price crosses below R1 with volume > 1.5x average and CHOP > 61.8.
Exit when price crosses the opposite level (S1 for longs, R1 for shorts) or CHOP < 38.2 (trending).
This strategy targets ranging markets where price respects pivot levels, avoiding trending regimes that cause whipsaws.
Designed for low frequency (12h) to minimize fee impact while capturing mean reversion in BTC/ETH ranging markets.
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
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (using previous day's data)
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    typical_range = high_1d - low_1d
    r1 = close_1d + typical_range * 1.1 / 12
    s1 = close_1d - typical_range * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate Choppiness Index on 12h data
    # CHOP = 100 * log10(sum(ATR)/ (max(high)-min(low))) / log10(period)
    chop_period = 14
    atr = np.full(n, np.nan)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    if n > 0:
        tr[0] = high[0] - low[0]  # First TR
    
    # Calculate ATR using Wilder's smoothing
    atr_period = chop_period
    if n >= atr_period:
        atr[atr_period-1] = np.mean(tr[:atr_period])
        for i in range(atr_period, n):
            atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Calculate Chopiness Index
    chop = np.full(n, np.nan)
    for i in range(chop_period, n):
        period_high = np.max(high[i-chop_period+1:i+1])
        period_low = np.min(low[i-chop_period+1:i+1])
        period_atr_sum = np.sum(atr[i-chop_period+1:i+1])
        if period_high > period_low and period_atr_sum > 0:
            chop[i] = 100 * np.log10(period_atr_sum / (period_high - period_low)) / np.log10(chop_period)
    
    # Volume confirmation: volume > 1.5x 24-period average (2 days)
    vol_ma = np.full(n, np.nan)
    vol_period = 24
    if n >= vol_period:
        for i in range(vol_period, n):
            vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, chop_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(chop[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Chop regime filter: only trade in ranging markets (CHOP > 61.8)
        in_range = chop[i] > 61.8
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0 and in_range:
            # Long: price crosses above S1 with volume confirmation
            if close[i] > s1_aligned[i] and close[i-1] <= s1_aligned[i-1] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below R1 with volume confirmation
            elif close[i] < r1_aligned[i] and close[i-1] >= r1_aligned[i-1] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below S1 OR chop < 38.2 (trending)
            if close[i] < s1_aligned[i] and close[i-1] >= s1_aligned[i-1]:
                signals[i] = -0.25  # reverse to short
                position = -1
            elif chop[i] < 38.2:
                signals[i] = 0.0  # go flat
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above R1 OR chop < 38.2 (trending)
            if close[i] > r1_aligned[i] and close[i-1] <= r1_aligned[i-1]:
                signals[i] = 0.25  # reverse to long
                position = 1
            elif chop[i] < 38.2:
                signals[i] = 0.0  # go flat
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_Regime_v2"
timeframe = "12h"
leverage = 1.0