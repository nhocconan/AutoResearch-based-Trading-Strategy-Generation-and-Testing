#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_Breakout_1dTrend_VolumeSpike
Hypothesis: Ichimoku cloud twist (Senkou Span A/B cross) from 1d timeframe as trend filter, combined with 6h price breaking above/below the cloud with volume confirmation (>2.0x 20-bar MA). Cloud twist indicates strong trend acceleration, providing edge in both bull and bear markets. Uses 6h timeframe for lower frequency (target: 12-30 trades/year) to minimize fee drag while capturing major moves. Works in ranging markets by requiring both cloud breakout and volume spike, reducing whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 52 for Senkou Span B (26*2)
        return np.zeros(n)
    
    # Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h timeframe (no additional delay needed for Senkou spans)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Cloud twist: Senkou A crossing above/below Senkou B (trend acceleration signal)
    # We use the previous value to detect the cross
    senkou_a_prev = np.roll(senkou_a_aligned, 1)
    senkou_b_prev = np.roll(senkou_b_aligned, 1)
    senkou_a_prev[0] = np.nan
    senkou_b_prev[0] = np.nan
    
    # Bullish twist: Senkou A crosses above Senkou B
    bullish_twist = (senkou_a_aligned > senkou_b_aligned) & (senkou_a_prev <= senkou_b_prev)
    # Bearish twist: Senkou A crosses below Senkou B
    bearish_twist = (senkou_a_aligned < senkou_b_aligned) & (senkou_a_prev >= senkou_b_prev)
    
    # The cloud boundaries (for price breakout)
    upper_cloud = np.maximum(senkou_a_aligned, senkou_b_aligned)
    lower_cloud = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size (25% of capital)
    
    # Warmup: max of calculations (52 for Senkou B, 20 for volume)
    start_idx = max(52, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(upper_cloud[i]) or 
            np.isnan(lower_cloud[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        bullish_twist_val = bullish_twist[i]
        bearish_twist_val = bearish_twist[i]
        vol_spike = volume_spike[i]
        
        # Entry conditions: price breaks cloud in direction of twist with volume spike
        long_entry = (close_val > upper_cloud[i]) and bullish_twist_val and vol_spike
        short_entry = (close_val < lower_cloud[i]) and bearish_twist_val and vol_spike
        
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
            # Long - exit when price re-enters cloud or twist reverses
            if close_val < lower_cloud[i] or not bullish_twist_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = base_size
        elif position == -1:
            # Short - exit when price re-enters cloud or twist reverses
            if close_val > upper_cloud[i] or not bearish_twist_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0