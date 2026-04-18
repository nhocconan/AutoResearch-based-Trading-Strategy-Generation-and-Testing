#!/usr/bin/env python3
"""
6h Weekly ICHIMOKU Cloud Breakout with Volume Confirmation
Hypothesis: ICHIMOKU cloud acts as dynamic support/resistance; weekly cloud breakouts capture major trend changes in both bull and bear markets. Volume confirmation filters false breaks. Target: 15-35 trades/year on 6f timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku Cloud components: tenkan, kijun, senkou_a, senkou_b"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2.0)
    
    return tenkan, kijun, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Ichimoku calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Ichimoku components on weekly data
    tenkan_1w, kijun_1w, senkou_a_1w, senkou_b_1w = calculate_ichimoku(high_1w, low_1w, close_1w)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_1w)
    kijun_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_1w)
    senkou_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_a_1w)
    senkou_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_1w)
    
    # Volume spike detection (1.5x 24-period average - 6 days worth)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100  # need enough history for Ichimoku calculations
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_1w_aligned[i]) or np.isnan(kijun_1w_aligned[i]) or 
            np.isnan(senkou_a_1w_aligned[i]) or np.isnan(senkou_b_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        tenkan = tenkan_1w_aligned[i]
        kijun = kijun_1w_aligned[i]
        senkou_a = senkou_a_1w_aligned[i]
        senkou_b = senkou_b_1w_aligned[i]
        
        # Determine cloud boundaries and color
        upper_cloud = max(senkou_a, senkou_b)
        lower_cloud = min(senkou_a, senkou_b)
        cloud_green = senkou_a > senkou_b  # bullish cloud
        
        if position == 0:
            # Long: price breaks above cloud with volume spike in bullish cloud
            if (price > upper_cloud and 
                volume_spike[i] and 
                cloud_green):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below cloud with volume spike in bearish cloud
            elif (price < lower_cloud and 
                  volume_spike[i] and 
                  not cloud_green):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price drops below cloud or Tenkan-Kijun cross down
            if price < lower_cloud or tenkan < kijun:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above cloud or Tenkan-Kijun cross up
            if price > upper_cloud or tenkan > kijun:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_IchimokuCloud_Breakout_Volume"
timeframe = "6h"
leverage = 1.0