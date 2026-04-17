#!/usr/bin/env python3
"""
6h_Ichimoku_KumoBreakout_VolumeFilter
Strategy: Ichimoku cloud breakout with volume confirmation on 6h timeframe.
Long: Price breaks above Kumo cloud (Senkou Span A/B) + volume > 1.5x avg + Tenkan > Kijun
Short: Price breaks below Kumo cloud + volume > 1.5x avg + Tenkan < Kijun
Exit: Price returns to Tenkan-sen line
Position size: 0.25
Designed to capture momentum breakouts with trend confirmation in both bull and bear markets.
Timeframe: 6h
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    # Not used in signals to avoid look-ahead
    
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Ichimoku on 6h data
    tenkan, kijun, senkou_a, senkou_b = calculate_ichimoku(high, low, close)
    
    # Kumo top and bottom (Senkou Span A and B)
    kumo_top = np.maximum(senkou_a, senkou_b)
    kumo_bottom = np.minimum(senkou_a, senkou_b)
    
    # Volume confirmation (20-period average)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 20)  # Need Senkou Span B data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Cloud breakout conditions
        price_above_kumo = close[i] > kumo_top[i]
        price_below_kumo = close[i] < kumo_bottom[i]
        
        # Trend confirmation: Tenkan vs Kijun
        tenkan_above_kijun = tenkan[i] > kijun[i]
        tenkan_below_kijun = tenkan[i] < kijun[i]
        
        # Re-entry condition: price returns to Tenkan-sen
        return_to_tenkan = abs(close[i] - tenkan[i]) < (0.003 * close[i])  # within 0.3%
        
        if position == 0:
            # Long: price breaks above cloud + volume + Tenkan > Kijun
            if price_above_kumo and volume_filter and tenkan_above_kijun:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below cloud + volume + Tenkan < Kijun
            elif price_below_kumo and volume_filter and tenkan_below_kijun:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to Tenkan or breaks below cloud
            if return_to_tenkan or price_below_kumo:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to Tenkan or breaks above cloud
            if return_to_tenkan or price_above_kumo:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_KumoBreakout_VolumeFilter"
timeframe = "6h"
leverage = 1.0