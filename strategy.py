#!/usr/bin/env python3
"""
6h_1d_Ichimoku_Cloud_Filter
Hypothesis: Use Ichimoku cloud from 1d as trend filter (price above/below cloud) and TK cross on 6h for entry.
In bull markets, buy when price above cloud and TK crosses up. In bear markets, sell when price below cloud and TK crosses down.
Volume confirmation ensures institutional participation. Target: 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_ktf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components."""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): close shifted 26 periods behind
    # Not used in this strategy
    
    return tenkan, kijun, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku on daily
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Cloud top and bottom (Senkou Span A and B)
    cloud_top = np.maximum(senkou_a_1d, senkou_b_1d)
    cloud_bottom = np.minimum(senkou_a_1d, senkou_b_1d)
    
    # Align Ichimoku components to 6h timeframe
    cloud_top_aligned = align_htf_to_ltf(prices, df_1d, cloud_top)
    cloud_bottom_aligned = align_htf_to_ltf(prices, df_1d, cloud_bottom)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    
    # TK cross on 6h: Tenkan crosses above/below Kijun
    # We need 6h Tenkan and Kijun for the cross
    tenkan_6h = pd.Series(close).rolling(window=9, min_periods=9).apply(
        lambda x: (x.max() + x.min()) / 2, raw=True
    ).values
    kijun_6h = pd.Series(close).rolling(window=26, min_periods=26).apply(
        lambda x: (x.max() + x.min()) / 2, raw=True
    ).values
    
    # TK cross signals (1 for bullish cross, -1 for bearish cross)
    tk_bullish = np.zeros(n, dtype=bool)
    tk_bearish = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if not (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
                np.isnan(tenkan_6h[i-1]) or np.isnan(kijun_6h[i-1])):
            tk_bullish[i] = (tenkan_6h[i-1] <= kijun_6h[i-1]) and (tenkan_6h[i] > kijun_6h[i])
            tk_bearish[i] = (tenkan_6h[i-1] >= kijun_6h[i-1]) and (tenkan_6h[i] < kijun_6h[i])
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(60, n):
        # Skip if any required data is not ready
        if (np.isnan(cloud_top_aligned[i]) or np.isnan(cloud_bottom_aligned[i]) or
            np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Price above cloud (bullish trend) or below cloud (bearish trend)
        price_above_cloud = close[i] > cloud_top_aligned[i]
        price_below_cloud = close[i] < cloud_bottom_aligned[i]
        
        # Long: price above cloud + TK bullish cross + volume expansion
        long_condition = price_above_cloud and tk_bullish[i] and volume_expansion[i]
        
        # Short: price below cloud + TK bearish cross + volume expansion
        short_condition = price_below_cloud and tk_bearish[i] and volume_expansion[i]
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif long_condition and position == 1:
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        elif short_condition and position == -1:
            signals[i] = -position_size
        else:
            # Exit conditions: reverse signal or price moves back into cloud
            if position == 1 and not (price_above_cloud or tk_bullish[i]):
                position = 0
                signals[i] = 0.0
            elif position == -1 and not (price_below_cloud or tk_bearish[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "6h_1d_Ichimoku_Cloud_Filter"
timeframe = "6h"
leverage = 1.0