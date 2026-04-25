#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend_VolumeConfirm
Hypothesis: On 6h timeframe, Ichimoku Kumo (cloud) twist from 1d timeframe signals major trend changes. 
A Kumo twist (Senkou Span A crossing Senkou Span B) on daily chart indicates institutional order flow shifts.
We enter on 6h breakout of the twist level in the direction of the new trend, confirmed by volume spike (>2x 20-bar average).
The twist acts as a strong support/resistance level. Works in both bull and bear markets as it captures trend changes.
Designed for low trade frequency (10-25/year) to minimize fee drag in 6h timeframe.
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
    
    # Get 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    displacement = 26
    
    # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past 9 periods
    highest_high_9 = pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max().values
    lowest_low_9 = pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min().values
    tenkan_sen = (highest_high_9 + lowest_low_9) / 2
    
    # Calculate Kijun-sen (Base Line): (highest high + lowest low)/2 for past 26 periods
    highest_high_26 = pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max().values
    lowest_low_26 = pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min().values
    kijun_sen = (highest_high_26 + lowest_low_26) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 displaced forward 26 periods
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Calculate Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past 52 periods displaced forward 26 periods
    highest_high_52 = pd.Series(high_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max().values
    lowest_low_52 = pd.Series(low_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min().values
    senkou_span_b = ((highest_high_52 + lowest_low_52) / 2)
    
    # Align Ichimoku components to 6h timeframe (displacement handled by align_htf_to_ltf)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Kumo twist detection: Senkou Span A crosses Senkou Span B
    # We need previous and current values to detect crossover
    ss_a_prev = np.concatenate([[np.nan], senkou_span_a_aligned[:-1]])
    ss_b_prev = np.concatenate([[np.nan], senkou_span_b_aligned[:-1]])
    
    # Bullish twist: SSA crosses above SSB (previous SSA <= previous SSB and current SSA > current SSB)
    bullish_twist = (ss_a_prev <= ss_b_prev) & (senkou_span_a_aligned > senkou_span_b_aligned)
    # Bearish twist: SSA crosses below SSB (previous SSA >= previous SSB and current SSA < current SSB)
    bearish_twist = (ss_a_prev >= ss_b_prev) & (senkou_span_a_aligned < senkou_span_b_aligned)
    
    # The twist level acts as support/resistance - we use the midpoint of the cloud as reference
    kumou_level = (senkou_span_a_aligned + senkou_span_b_aligned) / 2
    
    # Volume average (20-period) for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    # Need enough data for Ichimoku (52 periods) + displacement (26) + volume MA (20)
    start_idx = max(52 + 26, 20)  # 78
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or 
            np.isnan(kumou_level[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        kumo_val = kumou_level[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current volume > 2.0x 20-period average
        volume_spike = vol_val > 2.0 * vol_ma_val
        
        if position == 0:
            # Look for entry signals: 6h price breaks above/below Kumo level in direction of twist
            # Bullish twist + price breaks above Kumo level + volume spike = long
            bullish_entry = bullish_twist[i] and (high_val > kumo_val) and volume_spike
            # Bearish twist + price breaks below Kumo level + volume spike = short
            bearish_entry = bearish_twist[i] and (low_val < kumo_val) and volume_spike
            
            if bullish_entry:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif bearish_entry:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Price breaks below Kumo level (invalidates bullish structure)
            if close_val < kumo_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Opposite twist occurs (potential trend change)
            elif bearish_twist[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Price breaks above Kumo level (invalidates bearish structure)
            if close_val > kumo_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Opposite twist occurs (potential trend change)
            elif bullish_twist[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0