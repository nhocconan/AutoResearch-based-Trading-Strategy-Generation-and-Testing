#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_Volume_1dTrend
Hypothesis: Combines 12h Camarilla R1/S1 breakout with volume confirmation and 1d EMA trend filter.
Uses 12h Camarilla pivot levels for entry, volume spike for confirmation, and 1d EMA for trend filtering.
Designed to work in both bull and bear markets by requiring alignment with higher timeframe trend.
Targets 12-37 trades/year (50-150 total over 4 years).
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
    
    # Get 12h data for Camarilla pivot levels
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Camarilla pivot levels (R1, S1)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    pivot = np.full(len(high_12h), np.nan)
    r1 = np.full(len(high_12h), np.nan)
    s1 = np.full(len(high_12h), np.nan)
    
    for i in range(len(high_12h)):
        if i == 0:
            # Use previous day's data for first bar (not available, so skip)
            continue
        # Use previous day's OHLC to calculate today's pivot levels
        phigh = high_12h[i-1]
        plow = low_12h[i-1]
        pclose = close_12h[i-1]
        
        pp = (phigh + plow + pclose) / 3.0
        r1[i] = pp + (phigh - plow) * 1.1 / 12.0
        s1[i] = pp - (phigh - plow) * 1.1 / 12.0
        pivot[i] = pp  # not used directly but calculated for completeness
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA (34-period)
    ema34 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema34[33] = np.mean(close_1d[0:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema34[i] = close_1d[i] * alpha + ema34[i-1] * (1 - alpha)
    
    # Volume spike: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    # Align 12h Camarilla levels to 4h timeframe (using 12h data for 12h timeframe)
    # Note: Since we're using 12h timeframe, we need to align 12h data to itself
    # For 12h timeframe, the alignment is straightforward (no interpolation needed)
    # We'll use the 12h data directly but ensure we don't use current bar's data
    r1_aligned = np.full(n, np.nan)
    s1_aligned = np.full(n, np.nan)
    
    # Map 12h indices to 12h timeframe indices
    # For 12h timeframe, each 12h bar corresponds to multiple 12h bars (trivial)
    # We need to shift by 1 to avoid look-ahead (use previous bar's levels)
    j = 0  # index in 12h array
    for i in range(n):
        # Advance j to the corresponding 12h bar
        while j < len(df_12h) and df_12h.iloc[j]['open_time'] <= prices.iloc[i]['open_time']:
            j += 1
        # Use previous 12h bar's data (j-1) to avoid look-ahead
        if j-1 >= 0 and j-1 < len(r1):
            r1_aligned[i] = r1[j-1]
            s1_aligned[i] = s1[j-1]
    
    # Align 1d EMA to 12h timeframe
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA and enough history
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1 with volume spike and 1d uptrend
            if (close[i] > r1_aligned[i] and vol_spike[i] and 
                close[i] > ema34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume spike and 1d downtrend
            elif (close[i] < s1_aligned[i] and vol_spike[i] and 
                  close[i] < ema34_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below S1 or 1d trend turns down
            if (close[i] < s1_aligned[i] or close[i] < ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above R1 or 1d trend turns up
            if (close[i] > r1_aligned[i] or close[i] > ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_Volume_1dTrend"
timeframe = "12h"
leverage = 1.0