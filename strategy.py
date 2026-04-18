#!/usr/bin/env python3
"""
6h Donchian(20) Breakout with Weekly Pivot Direction and Volume Spike
Hypothesis: Breakouts beyond 6h Donchian channels aligned with weekly pivot (R1/S1) direction,
confirmed by volume spikes, capture momentum in both bull and bear markets.
Weekly pivot provides long-term bias, Donchian breakouts provide entry timing,
and volume filters out false breakouts. Designed for 12-37 trades/year on 6h timeframe.
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
    
    # Get weekly data for pivot calculation (once before loop)
    df_w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points using previous week's data
    weekly_high = df_w['high']
    weekly_low = df_w['low']
    weekly_close = df_w['close']
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    
    # Use previous week's levels only
    r1_prev = r1.shift(1).values
    s1_prev = s1.shift(1).values
    
    # Align to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_w, r1_prev)
    s1_aligned = align_htf_to_ltf(prices, df_w, s1_prev)
    
    # Weekly pivot direction: above pivot = bullish bias, below = bearish bias
    pivot_val = ((weekly_high + weekly_low + weekly_close) / 3).shift(1).values
    pivot_aligned = align_htf_to_ltf(prices, df_w, pivot_val)
    weekly_bullish = close > pivot_aligned  # Price above weekly pivot = bullish bias
    
    # Donchian channel (20-period) on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(pivot_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        
        if position == 0:
            # Long: bullish bias + break above Donchian high + volume spike
            if weekly_bullish[i] and price > upper and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish bias + break below Donchian low + volume spike
            elif not weekly_bullish[i] and price < lower and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: price returns to Donchian low or loses bullish bias
            if price < lower or not weekly_bullish[i]:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: price returns to Donchian high or gains bullish bias
            if price > upper or weekly_bullish[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian_WeeklyPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0