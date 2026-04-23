#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with Tenkan/Kijun cross and 1d cloud filter.
- Primary timeframe: 6h, HTF: 1d for cloud (Senkou Span A/B) filter
- Long: Tenkan crosses above Kijun + price > Senkou Span A (1d cloud top) + volume > 1.5x 20-period avg
- Short: Tenkan crosses below Kijun + price < Senkou Span B (1d cloud bottom) + volume > 1.5x 20-period avg
- Exit: Tenkan/Kijun cross reverses OR price re-enters the cloud (between Senkou A/B)
- Uses Ichimoku for trend/momentum with cloud as dynamic support/resistance
- Target: 80-120 total trades over 4 years (20-30/year) on 6h timeframe
- Discrete position sizing: ±0.25
- BTC/ETH focus: requires HTF cloud alignment to avoid false signals in chop
- Works in bull markets (cloud as support) and bear markets (cloud as resistance)
- Uses mtf_data helper for proper HTF alignment without look-ahead
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
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2.0
    
    # Calculate 1d Ichimoku cloud for HTF filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d Tenkan and Kijun
    period9_high_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (period9_high_1d + period9_low_1d) / 2.0
    
    period26_high_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (period26_high_1d + period26_low_1d) / 2.0
    
    # 1d Senkou Span A and B
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2.0
    period52_high_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (period52_high_1d + period52_low_1d) / 2.0
    
    # Align 1d cloud to 6h timeframe
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(52, 20)  # Need 52 for Senkou B, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        # Tenkan/Kijun cross detection
        tenkan_prev = tenkan[i-1] if i > 0 else tenkan[i]
        kijun_prev = kijun[i-1] if i > 0 else kijun[i]
        tenkan_cross_above = tenkan[i] > kijun[i] and tenkan_prev <= kijun_prev
        tenkan_cross_below = tenkan[i] < kijun[i] and tenkan_prev >= kijun_prev
        
        if position == 0:
            # Long: Tenkan crosses above Kijun + price > 1d Senkou A (cloud top) + volume spike
            if (tenkan_cross_above and 
                close[i] > senkou_a_aligned[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: Tenkan crosses below Kijun + price < 1d Senkou B (cloud bottom) + volume spike
            elif (tenkan_cross_below and 
                  close[i] < senkou_b_aligned[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Tenkan/Kijun cross reverses OR price re-enters cloud
            if tenkan_cross_below or (close[i] <= senkou_a_aligned[i] and close[i] >= senkou_b_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Tenkan/Kijun cross reverses OR price re-enters cloud
            if tenkan_cross_above or (close[i] >= senkou_b_aligned[i] and close[i] <= senkou_a_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_1dCloud_Filter_VolumeSpike"
timeframe = "6h"
leverage = 1.0