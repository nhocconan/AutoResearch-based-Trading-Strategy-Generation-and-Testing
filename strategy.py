#!/usr/bin/env python3
"""
1D Weekly Pivot Reversal with Volume Spike
Uses weekly pivot levels (R1/S1) as reversal zones confirmed by volume spikes.
Designed for low trade frequency (target: 7-25 trades/year) with strong reversal edge in range-bound markets.
Works in both bull and bear markets by fading extremes at weekly pivot levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly pivot points (using previous week's data)
    pivot = (high_weekly[:-1] + low_weekly[:-1] + close_weekly[:-1]) / 3.0
    r1 = 2 * pivot - low_weekly[:-1]
    s1 = 2 * pivot - high_weekly[:-1]
    
    # Align weekly pivot levels to daily timeframe (with 1-week delay for completed bar)
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot, additional_delay_bars=0)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1, additional_delay_bars=0)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1, additional_delay_bars=0)
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 200  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        
        if position == 0:
            # Long reversal: price touches S1 with volume spike
            if abs(price - s1_val) < 0.001 * s1_val and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short reversal: price touches R1 with volume spike
            elif abs(price - r1_val) < 0.001 * r1_val and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit at pivot or opposite level
            if price >= pivot_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit at pivot or opposite level
            if price <= pivot_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1D_Weekly_Pivot_Reversal_Volume_Spike"
timeframe = "1d"
leverage = 1.0