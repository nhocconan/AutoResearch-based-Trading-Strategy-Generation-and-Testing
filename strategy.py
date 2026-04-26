#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend_VolumeConfirm_v1
Hypothesis: Ichimoku Kumo twist (Senkou Span A/B cross) on 1d as regime filter, combined with TK cross on 6h for entry and volume confirmation. 
In bull markets (price above Kumo), TK cross up = long; in bear markets (price below Kumo), TK cross down = short. 
Volume spike confirms momentum. Targets 60-120 total trades over 4 years (15-30/year).
Works in both bull (trend following) and bear (counter-trend via Kumo filter) regimes.
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
    
    # Load 1d data ONCE before loop for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Kumo twist: Senkou Span A crosses Senkou Span B
    # Bullish twist: Senkou A > Senkou B (after previously being below)
    # Bearish twist: Senkou A < Senkou B (after previously being above)
    senkou_a_shift = np.roll(senkou_a, 1)
    senkou_b_shift = np.roll(senkou_b, 1)
    senkou_a_shift[0] = np.nan
    senkou_b_shift[0] = np.nan
    
    bullish_twist = (senkou_a > senkou_b) & (senkou_a_shift <= senkou_b_shift)
    bearish_twist = (senkou_a < senkou_b) & (senkou_a_shift >= senkou_b_shift)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    bullish_twist_aligned = align_htf_to_ltf(prices, df_1d, bullish_twist.astype(float))
    bearish_twist_aligned = align_htf_to_ltf(prices, df_1d, bearish_twist.astype(float))
    
    # TK cross on 6h: Tenkan crosses Kijun
    tk_cross_up = (tenkan_aligned > kijun_aligned) & (np.roll(tenkan_aligned, 1) <= np.roll(kijun_aligned, 1))
    tk_cross_down = (tenkan_aligned < kijun_aligned) & (np.roll(tenkan_aligned, 1) >= np.roll(kijun_aligned, 1))
    # Handle first element
    tk_cross_up[0] = False
    tk_cross_down[0] = False
    
    # Volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for Ichimoku)
    start_idx = max(52, 26, 9, 20) + 26  # Ichimoku needs 52 + 26 shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or
            np.isnan(senkou_b_aligned[i]) or
            np.isnan(bullish_twist_aligned[i]) or
            np.isnan(bearish_twist_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Determine market regime from Kumo twist
        bullish_regime = bullish_twist_aligned[i] > 0.5
        bearish_regime = bearish_twist_aligned[i] > 0.5
        
        # Long logic: in bullish regime, TK cross up with volume spike
        if bullish_regime and tk_cross_up[i] and volume_spike[i]:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: in bearish regime, TK cross down with volume spike
        elif bearish_regime and tk_cross_down[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: opposite TK cross or regime change
        elif position == 1 and (tk_cross_down[i] or not bullish_regime):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (tk_cross_up[i] or not bearish_regime):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0