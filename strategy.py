#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_Breakout_1dTrend_VolumeFilter
Hypothesis: Ichimoku Kumo (cloud) twist (Senkou Span A/B cross) on daily timeframe signals major trend changes. On 6h timeframe, enter breakouts above/below Kumo edges in direction of daily Kumo twist with volume confirmation (>1.5x 20-bar MA). Works in bull/bear markets by following daily Ichimoku trend while using Kumo as dynamic support/resistance. Volume filter reduces whipsaws. Target: 15-35 trades/year (60-140 total over 4 years).
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
    
    # Load 1d data ONCE before loop for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 26*2 for Senkou Span B
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku calculations (standard parameters: 9, 26, 52)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                 pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2).shift(26)
    # Chikou Span (Lagging Span): close shifted 26 periods behind (not used for signals)
    
    # Kumo twist detection: Senkou A crosses Senkou B
    # Bullish twist: Senkou A crosses above Senkou B
    # Bearish twist: Senkou A crosses below Senkou B
    senkou_a_vals = senkou_a.values
    senkou_b_vals = senkou_b.values
    
    bullish_twist = (senkou_a_vals > senkou_b_vals) & (senkou_a_vals.shift(1) <= senkou_b_vals.shift(1))
    bearish_twist = (senkou_a_vals < senkou_b_vals) & (senkou_a_vals.shift(1) >= senkou_b_vals.shift(1))
    
    # Current Kumo edges (Senkou Span A and B)
    kumo_top = np.maximum(senkou_a_vals, senkou_b_vals)  # Upper cloud boundary
    kumo_bottom = np.minimum(senkou_a_vals, senkou_b_vals)  # Lower cloud boundary
    
    # Align Ichimoku components to 6h timeframe
    kumo_top_aligned = align_htf_to_ltf(prices, df_1d, kumo_top)
    kumo_bottom_aligned = align_htf_to_ltf(prices, df_1d, kumo_bottom)
    bullish_twist_aligned = align_htf_to_ltf(prices, df_1d, bullish_twist.astype(float))
    bearish_twist_aligned = align_htf_to_ltf(prices, df_1d, bearish_twist.astype(float))
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size (25% of capital)
    
    # Warmup: max of calculations (52 for Ichimoku, 20 for volume)
    start_idx = max(52, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kumo_top_aligned[i]) or 
            np.isnan(kumo_bottom_aligned[i]) or 
            np.isnan(bullish_twist_aligned[i]) or 
            np.isnan(bearish_twist_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        kumo_top_val = kumo_top_aligned[i]
        kumo_bottom_val = kumo_bottom_aligned[i]
        bullish_twist_val = bullish_twist_aligned[i] > 0.5
        bearish_twist_val = bearish_twist_aligned[i] > 0.5
        vol_spike = volume_spike[i]
        
        # Determine Kumo relationship
        price_above_kumo = close_val > kumo_top_val
        price_below_kumo = close_val < kumo_bottom_val
        
        # Entry conditions: breakout of Kumo in direction of daily Kumo twist with volume spike
        long_entry = price_above_kumo and bullish_twist_val and vol_spike
        short_entry = price_below_kumo and bearish_twist_val and vol_spike
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = base_size
                position = 1
            elif short_entry:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price re-enters Kumo or Kumo twist reverses
            if close_val < kumo_top_val or not bullish_twist_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = base_size
        elif position == -1:
            # Short - exit when price re-enters Kumo or Kumo twist reverses
            if close_val > kumo_bottom_val or not bearish_twist_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_Breakout_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0