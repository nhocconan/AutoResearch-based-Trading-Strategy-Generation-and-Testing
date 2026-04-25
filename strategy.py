#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend_VolumeConfirm
Hypothesis: 6h Ichimoku cloud breakout with 1d trend filter (price above/below 1d Kumo) and volume confirmation (>2.0x 20-bar avg). 
Enters long when price breaks above 6h Kumo cloud in 1d uptrend with volume spike, short when price breaks below 6h Kumo cloud in 1d downtrend with volume spike. 
Exits when price re-enters the 6h Kumo cloud or 1d trend reverses. 
Designed for 6h timeframe with ~20-40 trades/year, works in bull/bear by following 1d trend filter and cloud breakout logic.
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
    
    # 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d Ichimoku components for trend filter: Tenkan-sen (9), Kijun-sen (26), Senkou Span A/B (52 displacement)
    tenkan_1d = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    kijun_1d = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    senkou_span_a_1d = ((tenkan_1d + kijun_1d) / 2).shift(22)  # displaced 26 periods forward, but we use 22 for alignment
    senkou_span_b_1d = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2).shift(22)
    # Kumo top/bottom for 1d trend: price > max(Senkou A, Senkou B) = uptrend, price < min = downtrend
    kumo_top_1d = np.maximum(senkou_span_a_1d.values, senkou_span_b_1d.values)
    kumo_bottom_1d = np.minimum(senkou_span_a_1d.values, senkou_span_b_1d.values)
    kumo_top_1d_aligned = align_htf_to_ltf(prices, df_1d, kumo_top_1d)
    kumo_bottom_1d_aligned = align_htf_to_ltf(prices, df_1d, kumo_bottom_1d)
    
    # 6h Ichimoku components for entry signals
    tenkan_6h = (pd.Series(high).rolling(window=9, min_periods=9).max() + pd.Series(low).rolling(window=9, min_periods=9).min()) / 2
    kijun_6h = (pd.Series(high).rolling(window=26, min_periods=26).max() + pd.Series(low).rolling(window=26, min_periods=26).min()) / 2
    senkou_span_a_6h = ((tenkan_6h + kijun_6h) / 2).shift(22)
    senkou_span_b_6h = ((pd.Series(high).rolling(window=52, min_periods=52).max() + pd.Series(low).rolling(window=52, min_periods=52).min()) / 2).shift(22)
    # Kumo top/bottom for 6h: cloud boundaries
    kumo_top_6h = np.maximum(senkou_span_a_6h.values, senkou_span_b_6h.values)
    kumo_bottom_6h = np.minimum(senkou_span_a_6h.values, senkou_span_b_6h.values)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough data for Ichimoku calculations (52 for Senkou B)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kumo_top_1d_aligned[i]) or 
            np.isnan(kumo_bottom_1d_aligned[i]) or 
            np.isnan(kumo_top_6h[i]) or 
            np.isnan(kumo_bottom_6h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above 6h Kumo top in 1d uptrend (price > 1d Kumo top) with volume confirmation
            long_setup = (close[i] > kumo_top_6h[i]) and (close[i-1] <= kumo_top_6h[i-1]) and (close[i] > kumo_top_1d_aligned[i]) and volume_spike[i]
            # Short: price breaks below 6h Kumo bottom in 1d downtrend (price < 1d Kumo bottom) with volume confirmation
            short_setup = (close[i] < kumo_bottom_6h[i]) and (close[i-1] >= kumo_bottom_6h[i-1]) and (close[i] < kumo_bottom_1d_aligned[i]) and volume_spike[i]
            
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
            # Exit: price re-enters 6h Kumo cloud OR 1d trend turns down (price < 1d Kumo bottom)
            if (close[i] < kumo_top_6h[i]) or (close[i] < kumo_bottom_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters 6h Kumo cloud OR 1d trend turns up (price > 1d Kumo top)
            if (close[i] > kumo_bottom_6h[i]) or (close[i] > kumo_top_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0