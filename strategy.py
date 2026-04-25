#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1dTrend
Hypothesis: Trade 6h timeframe using Ichimoku Tenkan-Kijun cross on 6h for entry signals,
filtered by 1d cloud (price above/below Kumo) for trend alignment, and confirmed by
6h volume spike (>2.0x 20-bar MA). Enter long when TK cross bullish, price above 1d cloud,
and volume spike. Enter short when TK cross bearish, price below 1d cloud, and volume spike.
Exit on opposite TK cross or cloud violation. Uses discrete sizing 0.25. Targets 12-37 trades/year
on 6h timeframe. Works in bull/bear via 1d cloud filter and volume confirmation to avoid whipsaws.
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
    
    # Calculate Ichimoku components on 6h (primary timeframe)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Get 1d data for cloud (Kumo) filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Ichimoku cloud components
    period9_high_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (period9_high_1d + period9_low_1d) / 2
    
    period26_high_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (period26_high_1d + period26_low_1d) / 2
    
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    period52_high_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (period52_high_1d + period52_low_1d) / 2
    
    # 1d Kumo cloud edges (Senkou Span A and B)
    kumo_top_1d = np.maximum(senkou_a_1d, senkou_b_1d)
    kumo_bottom_1d = np.minimum(senkou_a_1d, senkou_b_1d)
    
    # Align 1d cloud to 6h timeframe (completed daily bar only)
    kumo_top_1d_aligned = align_htf_to_ltf(prices, df_1d, kumo_top_1d)
    kumo_bottom_1d_aligned = align_htf_to_ltf(prices, df_1d, kumo_bottom_1d)
    
    # Calculate 6h volume spike (>2.0x 20-bar MA)
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_6h = volume > (2.0 * vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Ichimoku (52) and volume MA (20)
    start_idx = max(52, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(kumo_top_1d_aligned[i]) or np.isnan(kumo_bottom_1d_aligned[i]) or
            np.isnan(volume_spike_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: bullish TK cross (Tenkan > Kijun), price above 1d cloud, volume spike
            tk_bullish = tenkan[i] > kijun[i]
            price_above_cloud = close[i] > kumo_top_1d_aligned[i]
            long_setup = tk_bullish and price_above_cloud and volume_spike_6h[i]
            
            # Short: bearish TK cross (Tenkan < Kijun), price below 1d cloud, volume spike
            tk_bearish = tenkan[i] < kijun[i]
            price_below_cloud = close[i] < kumo_bottom_1d_aligned[i]
            short_setup = tk_bearish and price_below_cloud and volume_spike_6h[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: bearish TK cross OR price breaks below 1d cloud bottom
            if (tenkan[i] < kijun[i]) or (close[i] < kumo_bottom_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: bullish TK cross OR price breaks above 1d cloud top
            if (tenkan[i] > kijun[i]) or (close[i] > kumo_top_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1dTrend"
timeframe = "6h"
leverage = 1.0