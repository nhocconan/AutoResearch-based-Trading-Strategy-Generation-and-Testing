#!/usr/bin/env python3
# 1d_donchian_1w_camarilla_v1
# Hypothesis: Daily Donchian(20) breakout with 1-week Camarilla H3/L3 filter and volume confirmation.
# Uses 1d timeframe to minimize trade frequency (~10-25 trades/year). Donchian captures trends,
# Weekly Camarilla H3/L3 acts as strong structural bias filter (only trade in direction of weekly extremes),
# Volume spike confirms participation. Designed to work in both bull and bear markets by avoiding
# counter-trend entries and reducing whipsaws via tight entry conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_1w_camarilla_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels (primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    donchian_upper_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian levels to 1d timeframe (completed daily candle only)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_1d)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_1d)
    
    # Get 1w HTF data ONCE before loop for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for weekly
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Camarilla levels for weekly (H3/L3 for strong bias filter)
    h3_1w = pivot_1w + (range_1w * 1.1 / 4)
    l3_1w = pivot_1w - (range_1w * 1.1 / 4)
    
    # Align Camarilla levels to 1d timeframe (completed weekly candle only)
    h3_1w_aligned = align_htf_to_ltf(prices, df_1w, h3_1w)
    l3_1w_aligned = align_htf_to_ltf(prices, df_1w, l3_1w)
    
    # Volume spike detection (20-period volume average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(h3_1w_aligned[i]) or np.isnan(l3_1w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 1d Donchian lower band
            if close[i] < donchian_lower_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 1d Donchian upper band
            if close[i] > donchian_upper_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above 1d Donchian upper, above 1w H3, with volume spike
            if (close[i] > donchian_upper_aligned[i]) and (close[i] > h3_1w_aligned[i]) and vol_spike[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below 1d Donchian lower, below 1w L3, with volume spike
            elif (close[i] < donchian_lower_aligned[i]) and (close[i] < l3_1w_aligned[i]) and vol_spike[i]:
                position = -1
                signals[i] = -0.25
    
    return signals