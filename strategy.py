#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend_VolumeConfirm
Hypothesis: 6h Ichimoku system with Tenkan-Kijun cross and price breaking above/below Kumo (cloud) from 1d timeframe, filtered by 1d trend (price vs Kumo) and volume confirmation (>1.5x 20-bar average). 
Enters long when Tenkan > Kijun AND price > Kumo (cloud top) in 1d bullish regime (price > Kumo) with volume spike. 
Enters short when Tenkan < Kijun AND price < Kumo (cloud bottom) in 1d bearish regime (price < Kumo) with volume spike. 
Exits on opposite Tenkan/Kijun cross or when price re-enters the Kumo. 
Designed for 6h timeframe with ~20-40 trades/year, works in bull/bear by following 1d Ichimoku trend filter.
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
    
    # 1d data for HTF Ichimoku trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Ichimoku calculations (standard periods: 9, 26, 52)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_9 = (pd.Series(high_1d).rolling(window=9, min_periods=9).max().values + 
                pd.Series(low_1d).rolling(window=9, min_periods=9).min().values) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_26 = (pd.Series(high_1d).rolling(window=26, min_periods=26).max().values + 
                pd.Series(low_1d).rolling(window=26, min_periods=26).min().values) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_9 + kijun_26) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max().values + 
                 pd.Series(low_1d).rolling(window=52, min_periods=52).min().values) / 2)
    # Kumo (cloud) boundaries: Senkou Span A and B
    # For trend detection: we need current cloud, so we shift the calculated Senkou spans back by 26 to align with current price
    # But for simplicity in trend filter, we'll use the current Senkou A/B values as the cloud top/bottom for the current period
    # Actually, Senkou spans are plotted 26 periods ahead, so the current cloud is from 26 periods ago
    # To get current cloud: Senkou A/B calculated 26 periods ago
    # We'll calculate the cloud values that are relevant for current price (shifted back)
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    # Fill the first 26 values with NaN since they don't have Senkou data yet
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Kumo top (Senkou A) and bottom (Senkou B)
    kumo_top = np.maximum(senkou_a_shifted, senkou_b_shifted)
    kumo_bottom = np.minimum(senkou_a_shifted, senkou_b_shifted)
    
    # Align 1d Ichimoku components to 6h timeframe
    tenkan_9_aligned = align_htf_to_ltf(prices, df_1d, tenkan_9)
    kijun_26_aligned = align_htf_to_ltf(prices, df_1d, kijun_26)
    kumo_top_aligned = align_htf_to_ltf(prices, df_1d, kumo_top)
    kumo_bottom_aligned = align_htf_to_ltf(prices, df_1d, kumo_bottom)
    
    # 6h Ichimoku calculations for entry signals
    # Tenkan-sen (9-period)
    tenkan_9_6h = (pd.Series(high).rolling(window=9, min_periods=9).max().values + 
                   pd.Series(low).rolling(window=9, min_periods=9).min().values) / 2
    # Kijun-sen (26-period)
    kijun_26_6h = (pd.Series(high).rolling(window=26, min_periods=26).max().values + 
                   pd.Series(low).rolling(window=26, min_periods=26).min().values) / 2
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough data for all calculations
    start_idx = max(26, 52, 9, 20)  # 52 for Senkou B, 26 for Kijun, 9 for Tenkan
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_9_aligned[i]) or np.isnan(kijun_26_aligned[i]) or 
            np.isnan(kumo_top_aligned[i]) or np.isnan(kumo_bottom_aligned[i]) or
            np.isnan(tenkan_9_6h[i]) or np.isnan(kijun_26_6h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Tenkan > Kijun (bullish momentum) AND price > Kumo (bullish trend) 
            #        in 1d bullish regime (price > Kumo) with volume confirmation
            tenkan_bullish = tenkan_9_6h[i] > kijun_26_6h[i]
            price_above_kumo = close[i] > kumo_top_aligned[i]
            price_above_kumo_1d = close_1d[i // 24] > kumo_top[i // 24] if i // 24 < len(close_1d) else False  # Simplified 1d trend check
            # Use aligned 1d Ichimoku for trend: price > Kumo top from 1d
            bullish_1d_regime = close[i] > kumo_top_aligned[i]  # Already aligned, so this is 1d trend via 6h price vs 1d cloud
            
            long_setup = tenkan_bullish and price_above_kumo and bullish_1d_regime and volume_spike[i]
            
            # Short: Tenkan < Kijun (bearish momentum) AND price < Kumo (bearish trend)
            #        in 1d bearish regime (price < Kumo) with volume confirmation
            tenkan_bearish = tenkan_9_6h[i] < kijun_26_6h[i]
            price_below_kumo = close[i] < kumo_bottom_aligned[i]
            bearish_1d_regime = close[i] < kumo_bottom_aligned[i]  # 1d bearish regime
            
            short_setup = tenkan_bearish and price_below_kumo and bearish_1d_regime and volume_spike[i]
            
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
            # Exit: Tenkan < Kijun (momentum shift) OR price re-enters Kumo (below cloud top)
            if (tenkan_9_6h[i] < kijun_26_6h[i]) or (close[i] < kumo_top_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: Tenkan > Kijun (momentum shift) OR price re-enters Kumo (above cloud bottom)
            if (tenkan_9_6h[i] > kijun_26_6h[i]) or (close[i] > kumo_bottom_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0