#!/usr/bin/env python3
"""
12h_1w_Camarilla_R1_S1_Breakout_Volume_Trend_Filter
Hypothesis: Uses weekly Camarilla pivot levels (R1/S1) from 1w timeframe. Enters long when price breaks above weekly R1 with 1w EMA34 > EMA89 and volume spike. Enters short when price breaks below weekly S1 with 1w EMA34 < EMA89 and volume spike. Weekly timeframe reduces trade frequency to target 12-37 trades/year. Volume confirmation ensures momentum. Designed to work in both bull and bear markets by using weekly trend filter and breakout logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Typical price for pivot
    typical_price_1w = (high_1w + low_1w + close_1w) / 3.0
    
    # Calculate Camarilla levels: R1, S1
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    r1_1w = close_1w + 1.1 * (high_1w - low_1w) / 12.0
    s1_1w = close_1w - 1.1 * (high_1w - low_1w) / 12.0
    
    # Align weekly levels to 12h timeframe (wait for weekly close)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Weekly EMA trend filter: EMA34 and EMA89
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89_1w = close_1w_series.ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align EMAs to 12h timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    ema89_1w_aligned = align_htf_to_ltf(prices, df_1w, ema89_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Warmup for weekly indicators
    
    for i in range(start_idx, n):
        if (np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(ema89_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above weekly R1 with uptrend and volume spike
            if close[i] > r1_1w_aligned[i] and ema34_1w_aligned[i] > ema89_1w_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly S1 with downtrend and volume spike
            elif close[i] < s1_1w_aligned[i] and ema34_1w_aligned[i] < ema89_1w_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below weekly S1 or trend weakens
            if close[i] < s1_1w_aligned[i] or ema34_1w_aligned[i] <= ema89_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above weekly R1 or trend weakens
            if close[i] > r1_1w_aligned[i] or ema34_1w_aligned[i] >= ema89_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1w_Camarilla_R1_S1_Breakout_Volume_Trend_Filter"
timeframe = "12h"
leverage = 1.0