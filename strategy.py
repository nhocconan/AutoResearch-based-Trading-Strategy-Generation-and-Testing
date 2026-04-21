#!/usr/bin/env python3
"""
1d_1w_Turtle_Soup_Reversal
Hypothesis: Use 1-week Donchian breakout reversal with volume confirmation on 1d.
When price breaks above 1-week high but closes below previous 1d close, short reversal.
When price breaks below 1-week low but closes above previous 1d close, long reversal.
This traps breakout traders and captures mean reversion in ranging markets.
Works in bull markets by selling false breakouts and in bear markets by buying false breakdowns.
Volume confirmation ensures genuine participation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data once for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 20-period Donchian channels on weekly
    high_20 = np.full_like(high_1w, np.nan)
    low_20 = np.full_like(low_1w, np.nan)
    
    for i in range(len(high_1w)):
        if i >= 19:
            high_20[i] = np.max(high_1w[i-19:i+1])
            low_20[i] = np.min(low_1w[i-19:i+1])
    
    # Shift to align with current week (breakout based on prior week)
    high_20 = np.roll(high_20, 1)
    low_20 = np.roll(low_20, 1)
    high_20[0] = np.nan
    low_20[0] = np.nan
    
    high_20_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        prev_close = prices['close'].iloc[i-1]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.3 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.3 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long reversal: price breaks below weekly low but closes above prev close (bullish engulfing)
            if price < low_20_aligned[i] and prev_close > low_20_aligned[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short reversal: price breaks above weekly high but closes below prev close (bearish engulfing)
            elif price > high_20_aligned[i] and prev_close < high_20_aligned[i] and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below weekly low or reversal signal
            if price < low_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above weekly high or reversal signal
            if price > high_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Turtle_Soup_Reversal"
timeframe = "1d"
leverage = 1.0