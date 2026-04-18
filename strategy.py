#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_Volume_TrendFilter
Hypothesis: Camarilla pivot level breaks (R1/S1) on 4h with volume confirmation and 12h EMA trend filter.
Works in both bull and bear markets by only taking trades in the direction of the 12h trend.
Targets low trade frequency (20-50/year) to minimize fee drag while capturing high-probability breakouts.
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
    
    # Get 4h data for Camarilla pivot calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for the most recent completed 4h bar
    # Using previous bar's high/low/close to avoid look-ahead
    high_prev = np.roll(high_4h, 1)
    low_prev = np.roll(low_4h, 1)
    close_prev = np.roll(close_4h, 1)
    # First bar has no previous, fill with current
    high_prev[0] = high_4h[0]
    low_prev[0] = low_4h[0]
    close_prev[0] = close_4h[0]
    
    # Camarilla formulas
    R1 = close_prev + (high_prev - low_prev) * 1.1 / 12
    S1 = close_prev - (high_prev - low_prev) * 1.1 / 12
    
    # Align to 15m timeframe (4h -> 15m: 4*60/15 = 16 bars per 4h)
    R1_aligned = align_htf_to_ltf(prices, df_4h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_4h, S1)
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume spike: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20)  # Warmup for EMA and volume
    
    for i in range(start_idx, n):
        if (np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = R1_aligned[i]
        s1 = S1_aligned[i]
        ema34 = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and uptrend (price > EMA34)
            if price > r1 and vol_spike and price > ema34:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike and downtrend (price < EMA34)
            elif price < s1 and vol_spike and price < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price < R1 level OR trend turns down
            if price < r1 or price < ema34:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price > S1 level OR trend turns up
            if price > s1 or price > ema34:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_Volume_TrendFilter"
timeframe = "4h"
leverage = 1.0