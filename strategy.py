#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_TK_Cross_v1
Hypothesis: On 6h timeframe, trade Ichimoku Tenkan-Kijun cross signals filtered by 1d cloud color and volume confirmation. 
Ichimoku provides dynamic support/resistance and trend direction. The 1d cloud acts as a higher-timeframe regime filter: 
only take longs when price is above 1d cloud (bullish regime) and shorts when below 1d cloud (bearish regime). 
Volume spike confirms institutional participation at the crossover. Designed for 50-150 total trades over 4 years (12-37/year) 
with discrete sizing (0.25) to minimize fee drag. Works in bull/bear markets via 1d cloud filter.
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
    
    # Get 1d data for Ichimoku components and cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 for Senkou Span B
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2.0
    
    # Current 6h price relative to 1d cloud (use Senkou Span A and B)
    # Cloud top = max(Senkou A, Senkou B), Cloud bottom = min(Senkou A, Senkou B)
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    
    # Volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Align HTF indicators to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    cloud_top_aligned = align_htf_to_ltf(prices, df_1d, cloud_top)
    cloud_bottom_aligned = align_htf_to_ltf(prices, df_1d, cloud_bottom)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of Ichimoku periods (52) and volume MA (20)
    start_idx = max(52, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or
            np.isnan(cloud_top_aligned[i]) or
            np.isnan(cloud_bottom_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol_spike = volume_spike[i]
        cloud_top_val = cloud_top_aligned[i]
        cloud_bottom_val = cloud_bottom_aligned[i]
        
        # Determine cloud color: green if Senkou A > Senkou B (bullish), red otherwise
        # We need the raw Senkou A and B values to determine cloud color
        senkou_a_val = ((tenkan_val + kijun_val) / 2.0)  # Recalculate current Senkou A
        senkou_b_val = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max().iloc[-1] if len(high_1d) >= 52 else np.nan) + 
                       (pd.Series(low_1d).rolling(window=52, min_periods=52).min().iloc[-1] if len(low_1d) >= 52 else np.nan)) / 2.0
        # For simplicity, use aligned Senkou A and B from arrays (but we need to align them properly)
        # Instead, we'll calculate cloud color from the aligned Senkou Span A and B
        # We need to align Senkou A and B separately
        # Recompute aligned Senkou A and B
        senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
        senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        cloud_bullish = senkou_a_val > senkou_b_val  # Green cloud
        
        if position == 0:
            # Long: Tenkan crosses above Kijun, price above cloud, bullish cloud, volume spike
            tk_cross_up = (tenkan_val > kijun_val) and (tenkan_aligned[i-1] <= kijun_aligned[i-1]) if i > 0 else False
            price_above_cloud = close_val > cloud_top_val
            long_signal = tk_cross_up and price_above_cloud and cloud_bullish and vol_spike
            
            # Short: Tenkan crosses below Kijun, price below cloud, bearish cloud, volume spike
            tk_cross_down = (tenkan_val < kijun_val) and (tenkan_aligned[i-1] >= kijun_aligned[i-1]) if i > 0 else False
            price_below_cloud = close_val < cloud_bottom_val
            short_signal = tk_cross_down and price_below_cloud and (not cloud_bullish) and vol_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Tenkan crosses below Kijun OR price breaks below cloud bottom
            tk_cross_down = (tenkan_val < kijun_val) and (tenkan_aligned[i-1] >= kijun_aligned[i-1]) if i > 0 else False
            price_below_cloud = close_val < cloud_bottom_val
            if tk_cross_down or price_below_cloud:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Tenkan crosses above Kijun OR price breaks above cloud top
            tk_cross_up = (tenkan_val > kijun_val) and (tenkan_aligned[i-1] <= kijun_aligned[i-1]) if i > 0 else False
            price_above_cloud = close_val > cloud_top_val
            if tk_cross_up or price_above_cloud:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_v1"
timeframe = "6h"
leverage = 1.0