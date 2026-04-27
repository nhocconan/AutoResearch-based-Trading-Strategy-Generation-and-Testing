#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_Trend_Continuation
Hypothesis: Uses Ichimoku cloud twist (Senkou Span A/B cross) on 1d timeframe as trend filter for 6h breakout entries.
Enter long when price breaks above 6h Donchian(20) high AND 1d cloud is bullish (Senkou A > Senkou B) AND price > 1d Kumo.
Enter short when price breaks below 6h Donchian(20) low AND 1d cloud is bearish (Senkou A < Senkou B) AND price < 1d Kumo.
Exit when price returns to 6h Donchian(20) midpoint OR cloud twists opposite.
Ichimoku cloud twist confirms higher timeframe trend change with minimal lag. Donchian breakout provides clean entry.
Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position size. Works in bull/bear via cloud filter.
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
    
    # Get 1d and 6h data
    df_1d = get_htf_data(prices, '1d')
    df_6h = get_htf_data(prices, '6h')
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (high_tenkan + low_tenkan) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (high_kijun + low_kijun) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2.0)
    # Senkou Span B (Leading Span B): (52-period high + low) / 2 shifted 26 periods ahead
    period_senkou_b = 52
    high_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((high_senkou_b + low_senkou_b) / 2.0)
    
    # Current Kumo (cloud) boundaries: Senkou Span A/B shifted back 26 periods
    # For current cloud, we need values that were plotted 26 periods ago
    senkou_a_lagged = np.roll(senkou_a, 26)
    senkou_b_lagged = np.roll(senkou_b, 26)
    # First 26 values invalid
    senkou_a_lagged[:26] = np.nan
    senkou_b_lagged[:26] = np.nan
    
    # Align 1d Ichimoku to 6h timeframe
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_lagged)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_lagged)
    
    # 6h Donchian(20) for breakout signals
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    donchian_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Align 6h Donchian to 6h timeframe (no additional alignment needed as already 6h)
    donchian_high_aligned = donchian_high  # Already 6h data
    donchian_low_aligned = donchian_low
    donchian_mid_aligned = donchian_mid
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need Ichimoku (52), Donchian (20)
    start_idx = max(52, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        donchian_high_val = donchian_high_aligned[i]
        donchian_low_val = donchian_low_aligned[i]
        donchian_mid_val = donchian_mid_aligned[i]
        
        # Determine cloud state
        cloud_bullish = senkou_a_val > senkou_b_val
        cloud_bearish = senkou_a_val < senkou_b_val
        price_above_cloud = close_val > max(senkou_a_val, senkou_b_val)
        price_below_cloud = close_val < min(senkou_a_val, senkou_b_val)
        
        if position == 0:
            # Look for entry: Donchian breakout with cloud alignment
            # Long: price breaks above Donchian high AND bullish cloud AND price above cloud
            long_condition = (close_val > donchian_high_val) and cloud_bullish and price_above_cloud
            # Short: price breaks below Donchian low AND bearish cloud AND price below cloud
            short_condition = (close_val < donchian_low_val) and cloud_bearish and price_below_cloud
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price returns to Donchian midpoint OR cloud turns bearish
            exit_condition = (close_val <= donchian_mid_val) or (not cloud_bullish)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price returns to Donchian midpoint OR cloud turns bullish
            exit_condition = (close_val >= donchian_mid_val) or (not cloud_bearish)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_Trend_Continuation"
timeframe = "6h"
leverage = 1.0