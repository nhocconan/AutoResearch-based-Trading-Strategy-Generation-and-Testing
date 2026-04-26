#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend_VolumeFilter_v2
Hypothesis: Uses Ichimoku cloud twist (Senkou Span A/B cross) from 1d timeframe as trend filter, combined with 6h Tenkan/Kijun cross for entry and volume confirmation. Works in bull/bear markets by using 1d cloud direction for trend and volume spike to filter false signals. Targets 15-25 trades/year to minimize fee drag. Uses discrete position sizing (0.25) to reduce churn.
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
    
    # Get 1d data for Ichimoku (cloud twist and trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 26*2 for Senkou Span
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    tenkan_1d = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                 pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + low) / 2
    kijun_1d = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2).shift(2)  # Shifted 2 periods ahead
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    senkou_b_1d = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                    pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2).shift(2)
    
    # Cloud twist: Senkou A crosses above/below Senkou B
    # Twist up (bullish): Senkou A > Senkou B after previously being below
    senkou_a_vals = senkou_a_1d.values
    senkou_b_vals = senkou_b_1d.values
    twist_up = (senkou_a_vals > senkou_b_vals) & (np.roll(senkou_a_vals, 1) <= np.roll(senkou_b_vals, 1))
    # Twist down (bearish): Senkou A < Senkou B after previously being above
    twist_down = (senkou_a_vals < senkou_b_vals) & (np.roll(senkou_a_vals, 1) >= np.roll(senkou_b_vals, 1))
    
    # Align 1d Ichimoku components to 6h timeframe
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_vals)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_vals)
    twist_up_aligned = align_htf_to_ltf(prices, df_1d, twist_up.astype(float))
    twist_down_aligned = align_htf_to_ltf(prices, df_1d, twist_down.astype(float))
    
    # Calculate 6h Tenkan and Kijun for entry signals
    tenkan_6h = (pd.Series(high).rolling(window=9, min_periods=9).max() + 
                 pd.Series(low).rolling(window=9, min_periods=9).min()) / 2
    kijun_6h = (pd.Series(high).rolling(window=26, min_periods=26).max() + 
                pd.Series(low).rolling(window=26, min_periods=26).min()) / 2
    
    # TK Cross: Tenkan crosses above/below Kijun
    tk_cross_up = (tenkan_6h > kijun_6h) & (np.roll(tenkan_6h, 1) <= np.roll(kijun_6h, 1))
    tk_cross_down = (tenkan_6h < kijun_6h) & (np.roll(tenkan_6h, 1) >= np.roll(kijun_6h, 1))
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough for all indicators (52 for Senkou B, 26 for Kijun/Tenkan, 20 for volume)
    start_idx = max(52, 26, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(twist_up_aligned[i]) or np.isnan(twist_down_aligned[i]) or
            np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend from 1d cloud twist
        bullish_trend = twist_up_aligned[i] > 0.5
        bearish_trend = twist_down_aligned[i] > 0.5
        
        if position == 0:
            # Flat - look for TK cross with trend and volume confirmation
            # Long: Tenkan crosses above Kijun + bullish 1d cloud twist + volume spike
            long_entry = tk_cross_up[i] and bullish_trend and volume_spike[i]
            # Short: Tenkan crosses below Kijun + bearish 1d cloud twist + volume spike
            short_entry = tk_cross_down[i] and bearish_trend and volume_spike[i]
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on TK cross down or bearish twist
            exit_condition = tk_cross_down[i] or bearish_trend
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short - exit on TK cross up or bullish twist
            exit_condition = tk_cross_up[i] or bullish_trend
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrend_VolumeFilter_v2"
timeframe = "6h"
leverage = 1.0