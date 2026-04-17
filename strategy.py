#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 12h Ichimoku Cloud as trend filter and 1d Camarilla pivot R2/S2 for breakout entries.
Long when price breaks above 1d Camarilla R2 with volume confirmation and price > 12h Ichimoku Cloud (bullish regime).
Short when price breaks below 1d Camarilla S2 with volume confirmation and price < 12h Ichimoku Cloud (bearish regime).
Ichimoku Cloud from 12h provides multi-timeframe trend alignment, reducing false breakouts in ranging markets.
Camarilla R2/S2 levels offer stronger support/resistance than R1/S1, increasing breakout validity.
Designed to capture momentum shifts in both bull and bear markets by requiring alignment with higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels (R2, S2)
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R2 = C + Range * 1.1 / 6
    # S2 = C - Range * 1.1 / 6
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r2_1d = close_1d + range_1d * 1.1 / 6.0
    s2_1d = close_1d - range_1d * 1.1 / 6.0
    
    # Get 12h data for Ichimoku Cloud
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Ichimoku Cloud components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high_12h).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_12h).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high_12h).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_12h).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high_12h).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_12h).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2.0
    
    # The Cloud: between Senkou Span A and B
    # For trend filter: price above cloud = bullish, price below cloud = bearish
    # We'll use the cloud's leading edge (shifted 26 periods ahead) for actual cloud position
    # But for simplicity in HTF alignment, we use current cloud and align it
    ichimoku_top = np.maximum(senkou_a, senkou_b)
    ichimoku_bottom = np.minimum(senkou_a, senkou_b)
    
    # Calculate 6h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h timeframe
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    ichimoku_top_aligned = align_htf_to_ltf(prices, df_12h, ichimoku_top)
    ichimoku_bottom_aligned = align_htf_to_ltf(prices, df_12h, ichimoku_bottom)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for Ichimoku calculations (52+26)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r2_1d_aligned[i]) or 
            np.isnan(s2_1d_aligned[i]) or 
            np.isnan(ichimoku_top_aligned[i]) or 
            np.isnan(ichimoku_bottom_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above 1d Camarilla R2 with volume and bullish cloud (price > ichimoku top)
            if (close[i] > r2_1d_aligned[i] and 
                volume_confirmed and 
                close[i] > ichimoku_top_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Camarilla S2 with volume and bearish cloud (price < ichimoku bottom)
            elif (close[i] < s2_1d_aligned[i] and 
                  volume_confirmed and 
                  close[i] < ichimoku_bottom_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 1d Camarilla S2 (opposite side)
            if close[i] < s2_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 1d Camarilla R2 (opposite side)
            if close[i] > r2_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_12hIchimoku_CloudFilter_1dCamarilla_R2S2_Breakout_Volume"
timeframe = "6h"
leverage = 1.0