#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation
# Uses weekly pivot levels (from 1w data) to determine market bias: only long when price > weekly pivot, short when price < weekly pivot
# Donchian breakouts provide entry signals in the direction of weekly bias
# Volume confirmation ensures breakouts have participation
# Works in both bull/bear: weekly pivot adapts to long-term trend, Donchian captures breakouts in ranging markets
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag

name = "6h_1w_donchian_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot levels (standard floor pivot) using previous week's data
    weekly_pivot = np.full(len(df_1w), np.nan)
    weekly_r1 = np.full(len(df_1w), np.nan)
    weekly_s1 = np.full(len(df_1w), np.nan)
    
    for i in range(len(df_1w)):
        if i < 1:
            weekly_pivot[i] = np.nan
            weekly_r1[i] = np.nan
            weekly_s1[i] = np.nan
        else:
            # Use previous week's OHLC to calculate current week's pivot levels
            prev_high = df_1w['high'].iloc[i-1]
            prev_low = df_1w['low'].iloc[i-1]
            prev_close = df_1w['close'].iloc[i-1]
            
            pivot = (prev_high + prev_low + prev_close) / 3.0
            weekly_pivot[i] = pivot
            weekly_r1[i] = 2 * pivot - prev_low
            weekly_s1[i] = 2 * pivot - prev_high
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_6h = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_6h = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_6h = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Calculate 20-period Donchian channels on 6h
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(n):
        if i < 20:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(weekly_pivot_6h[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirm = volume[i] > 1.3 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR weekly bias turns bearish (price < weekly pivot)
            if close[i] < donchian_low[i] or close[i] < weekly_pivot_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR weekly bias turns bullish (price > weekly pivot)
            if close[i] > donchian_high[i] or close[i] > weekly_pivot_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Donchian breakout in direction of weekly bias with volume confirmation
            if volume_confirm:
                # Long breakout: price closes above Donchian high AND price > weekly pivot (bullish bias)
                if close[i] > donchian_high[i] and close[i] > weekly_pivot_6h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price closes below Donchian low AND price < weekly pivot (bearish bias)
                elif close[i] < donchian_low[i] and close[i] < weekly_pivot_6h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals