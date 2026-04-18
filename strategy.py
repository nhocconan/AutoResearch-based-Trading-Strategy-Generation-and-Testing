#!/usr/bin/env python3
"""
6h Ichimoku Cloud + 1d VWAP Pullback with Volume Confirmation
Long: Price above Kumo (cloud) + Tenkan > Kijun + price pulls back to Tenkan/Kijun + volume spike
Short: Price below Kumo + Tenkan < Kijun + price pulls back to Tenkan/Kijun + volume spike
Exit: Price crosses opposite edge of Kumo or Tenkan/Kijun cross reverses
Ichimoku provides dynamic support/resistance and trend direction. Pullback to Tenkan/Kijun in trending markets offers high-probability entries.
1d VWAP acts as institutional reference point for pullback depth. Volume spike confirms institutional participation.
Target: 60-120 total trades over 4 years (15-30/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: Tenkan, Kijun, Senkou A/B, Chikou"""
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    tenkan = (pd.Series(high).rolling(window=9, min_periods=9).max() + 
              pd.Series(low).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + low)/2
    kijun = (pd.Series(high).rolling(window=26, min_periods=26).max() + 
             pd.Series(low).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 52 periods
    senkou_b = ((pd.Series(high).rolling(window=52, min_periods=52).max() + 
                 pd.Series(low).rolling(window=52, min_periods=52).min()) / 2).shift(52)
    return tenkan, kijun, senkou_a, senkou_b

def calculate_vwap(high, low, close, volume):
    """Calculate Volume Weighted Average Price"""
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    return vwap

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku on 6h
    tenkan, kijun, senkou_a, senkou_b = calculate_ichimoku(high, low, close)
    
    # Get 1d data for VWAP
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d VWAP
    vwap_1d = calculate_vwap(high_1d, low_1d, close_1d, volume_1d)
    
    # Align 1d VWAP to 6h
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 52  # need Ichimoku calculations
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(vwap_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        # Determine if price is above or below cloud
        above_cloud = price > max(senkou_a[i], senkou_b[i])
        below_cloud = price < min(senkou_a[i], senkou_b[i])
        
        if position == 0:
            # Long: Above cloud + Tenkan > Kijun + pullback to Tenkan/Kijun + volume spike
            if (above_cloud and tenkan[i] > kijun[i] and
                price >= min(tenkan[i], kijun[i]) * 0.998 and  # within 0.2% of Tenkan/Kijun
                price <= max(tenkan[i], kijun[i]) * 1.002 and
                volume[i] > 1.5 * pd.Series(volume).rolling(window=20, min_periods=20).mean().iloc[i]):
                signals[i] = 0.25
                position = 1
            # Short: Below cloud + Tenkan < Kijun + pullback to Tenkan/Kijun + volume spike
            elif (below_cloud and tenkan[i] < kijun[i] and
                  price >= min(tenkan[i], kijun[i]) * 0.998 and
                  price <= max(tenkan[i], kijun[i]) * 1.002 and
                  volume[i] > 1.5 * pd.Series(volume).rolling(window=20, min_periods=20).mean().iloc[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below cloud OR Tenkan/Kijun cross turns bearish
            if (below_cloud or tenkan[i] < kijun[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above cloud OR Tenkan/Kijun cross turns bullish
            if (above_cloud or tenkan[i] > kijun[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_1dVWAP_Pullback_Volume"
timeframe = "6h"
leverage = 1.0