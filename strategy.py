#!/usr/bin/env python3
"""
6h_1w_Ichimoku_Cloud_Filter
Hypothesis: Use weekly Ichimoku cloud as trend filter and Tenkan/Kijun cross for entry on 6h.
In bull markets: price above cloud + TK cross up = long.
In bear markets: price below cloud + TK cross down = short.
Requires volume confirmation to avoid false signals.
Target: 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: Tenkan, Kijun, Senkou A/B, Chikou."""
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 52 periods
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): close shifted -22 periods (not used for forward signals)
    return tenkan, kijun, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Ichimoku trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Ichimoku on weekly
    tenkan_1w, kijun_1w, senkou_a_1w, senkou_b_1w = calculate_ichimoku(high_1w, low_1w, close_1w)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_1w)
    kijun_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_1w)
    senkou_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_a_1w)
    senkou_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_1w)
    
    # Cloud top and bottom (Senkou A/B)
    cloud_top = np.maximum(senkou_a_1w_aligned, senkou_b_1w_aligned)
    cloud_bottom = np.minimum(senkou_a_1w_aligned, senkou_b_1w_aligned)
    
    # TK cross signals
    tk_cross_up = tenkan_1w_aligned > kijun_1w_aligned
    tk_cross_down = tenkan_1w_aligned < kijun_1w_aligned
    
    # Price vs cloud
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(60, n):  # Start after Ichimoku warmup
        # Skip if any required data is not ready
        if (np.isnan(tenkan_1w_aligned[i]) or np.isnan(kijun_1w_aligned[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(tk_cross_up[i]) or np.isnan(tk_cross_down[i]) or
            np.isnan(price_above_cloud[i]) or np.isnan(price_below_cloud[i]) or
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long: price above cloud + TK cross up + volume expansion
        long_condition = price_above_cloud[i] and tk_cross_up[i] and volume_expansion[i]
        
        # Short: price below cloud + TK cross down + volume expansion
        short_condition = price_below_cloud[i] and tk_cross_down[i] and volume_expansion[i]
        
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
            # Exit: reverse signal or loss of cloud alignment
            if position == 1 and (not price_above_cloud[i] or not tk_cross_up[i]):
                position = 0
                signals[i] = 0.0
            elif position == -1 and (not price_below_cloud[i] or not tk_cross_down[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "6h_1w_Ichimoku_Cloud_Filter"
timeframe = "6h"
leverage = 1.0