#!/usr/bin/env python3
"""
6h Camarilla H3/L3 Breakout with Weekly Pivot Direction Filter
Hypothesis: Camarilla H3/L3 breakouts on 6h timeframe, when aligned with the weekly
pivot trend (price above/below weekly pivot), capture institutional momentum with
reduced false signals. The weekly pivot acts as a higher-timeframe bias filter,
improving performance in both bull and bear markets by ensuring trades align
with the dominant weekly trend. Volume confirmation is added to validate breakout
strength. Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
by requiring confluence of Camarilla breakout, weekly pivot trend, and volume spike.
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
    
    # Load weekly data ONCE before loop for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard: P = (H+L+C)/3)
    # Using previous completed weekly bar
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    weekly_pivot = (h_1w + l_1w + c_1w) / 3.0
    
    # Align weekly pivot to 6h (no extra delay as based on completed weekly bar)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Load 1d data for Camarilla pivot calculation (more responsive than 12h)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels on 1d: based on previous 1d bar's H, L, C
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    camarilla_h3 = c_1d + 1.1 * (h_1d - l_1d) / 2
    camarilla_l3 = c_1d - 1.1 * (h_1d - l_1d) / 2
    
    # Align Camarilla levels to 6h
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 20)  # weekly pivot needs 1 bar, volume MA 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to weekly pivot
        bullish_bias = curr_close > weekly_pivot_aligned[i]
        bearish_bias = curr_close < weekly_pivot_aligned[i]
        
        if position == 0:
            # Look for entry signals - require ALL conditions: Camarilla breakout + weekly trend + volume
            # Long: price breaks above Camarilla H3 AND bullish bias AND volume spike
            long_entry = (curr_high > camarilla_h3_aligned[i]) and bullish_bias and vol_spike
            # Short: price breaks below Camarilla L3 AND bearish bias AND volume spike
            short_entry = (curr_low < camarilla_l3_aligned[i]) and bearish_bias and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below Camarilla L3 (mean reversion) OR loss of bullish bias (below weekly pivot)
            if (curr_low < camarilla_l3_aligned[i]) or (curr_close < weekly_pivot_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above Camarilla H3 (mean reversion) OR loss of bearish bias (above weekly pivot)
            if (curr_high > camarilla_h3_aligned[i]) or (curr_close > weekly_pivot_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Breakout_WeeklyPivot_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0