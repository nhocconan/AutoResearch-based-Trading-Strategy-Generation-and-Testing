#!/usr/bin/env python3
"""
12h 1D High/Low Breakout with Volume and Trend Filter
Hypothesis: Price breaking above the prior day's high or below the prior day's low
signifies institutional interest and momentum continuation. Volume confirms the breakout
strength, while a 1-day EMA trend filter ensures alignment with the higher timeframe trend.
This strategy targets medium-term moves in both bull and bear markets with low trade frequency.
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
    
    # Get 1D data for prior day's high/low and EMA (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior day's high and low (using previous day's values)
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    
    # 1D EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1D indicators to 12H timeframe
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(prev_high_aligned[i]) or np.isnan(prev_low_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_high = high[i] > prev_high_aligned[i]
        breakout_low = low[i] < prev_low_aligned[i]
        vol_ok = vol_spike[i]
        trend = ema34_1d_aligned[i]
        
        if position == 0:
            # Enter long on upward breakout with volume and uptrend
            if breakout_high and vol_ok and close[i] > trend:
                signals[i] = 0.25
                position = 1
            # Enter short on downward breakout with volume and downtrend
            elif breakout_low and vol_ok and close[i] < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long on downward breakdown or trend change
            if low[i] < prev_low_aligned[i] or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short on upward breakout or trend change
            if high[i] > prev_high_aligned[i] or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1D_High_Low_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0