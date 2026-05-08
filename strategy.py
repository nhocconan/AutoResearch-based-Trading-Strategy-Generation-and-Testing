#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter with Donchian(20) breakout
# Uses daily Choppiness Index to filter trending vs ranging markets.
# In trending (CHOP < 38.2): follow Donchian breakouts.
# In ranging (CHOP > 61.8): mean-revert at Donchian channels.
# Combines regime detection with price channels for robust performance in both bull/bear markets.
# Target: <150 total trades to minimize fee drag.

name = "4h_Chop_Donchian_Breakout_Rev"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Choppiness Index calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First period
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ATR(14) for 1d
    atr_1d = np.zeros_like(tr)
    atr_1d[0] = tr[0]
    for i in range(1, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14  # Wilder's smoothing
    
    # Calculate Choppiness Index (14-period)
    chop = np.full_like(close_1d, 50.0)  # Default neutral
    for i in range(14, len(close_1d)):
        atr_sum = np.sum(atr_1d[i-13:i+1])
        highest_high = np.max(high_1d[i-13:i+1])
        lowest_low = np.min(low_1d[i-13:i+1])
        if highest_high != lowest_low:
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    
    # Align Choppiness Index to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian channels (20-period) on 4h
    highest_high = np.zeros_like(high)
    lowest_low = np.zeros_like(low)
    for i in range(20, n):
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    # Fill initial values
    for i in range(20):
        highest_high[i] = high[i]
        lowest_low[i] = low[i]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Ensure Donchian and Chop have data
    
    for i in range(start_idx, n):
        if np.isnan(chop_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop_aligned[i]
        
        if position == 0:
            # Determine regime and enter accordingly
            if chop_val < 38.2:  # Trending market
                # Enter long on breakout above upper band
                if close[i] > highest_high[i]:
                    signals[i] = 0.25
                    position = 1
                # Enter short on breakdown below lower band
                elif close[i] < lowest_low[i]:
                    signals[i] = -0.25
                    position = -1
            elif chop_val > 61.8:  # Ranging market
                # Mean reversion: buy at support, sell at resistance
                if close[i] <= lowest_low[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= highest_high[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long conditions
            if chop_val < 38.2:  # Trending: exit on breakdown
                if close[i] < lowest_low[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Ranging: exit at resistance or middle
                if close[i] >= highest_high[i] or close[i] >= (highest_high[i] + lowest_low[i]) / 2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Exit short conditions
            if chop_val < 38.2:  # Trending: exit on breakout
                if close[i] > highest_high[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # Ranging: exit at support or middle
                if close[i] <= lowest_low[i] or close[i] <= (highest_high[i] + lowest_low[i]) / 2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals