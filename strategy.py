#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_1dCloudFilter
Hypothesis: On 6h timeframe, Ichimoku Tenkan-Kijun cross with 1d cloud filter provides high-probability entries.
Trend direction determined by 1d cloud color (green=uptrend, red=downtrend). TK cross acts as momentum entry signal.
Volume confirmation (>1.5x 20-bar average) filters low-quality signals. Designed for 6h to capture multi-day moves
with tight stops via reverse TK cross. Target: 50-150 total trades over 4 years (12-37/year).
Works in both bull and bear markets by using 1d cloud as regime filter and TK cross for timely entries/exits.
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
    
    # Get 1d data for HTF Ichimoku cloud (Senkou Span A/B)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 26*2 for Senkou Span B
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on 1d
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
    
    # Align Ichimoku components to 6h timeframe (cloud is forward-looking, so we need to align properly)
    # For cloud, we want the values that were known at the time (no look-ahead)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen, additional_delay_bars=0)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen, additional_delay_bars=0)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a, additional_delay_bars=0)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b, additional_delay_bars=0)
    
    # Cloud color: green when Senkou A > Senkou B (uptrend), red when Senkou A < Senkou B (downtrend)
    cloud_green = senkou_a_aligned > senkou_b_aligned
    cloud_red = senkou_a_aligned < senkou_b_aligned
    
    # TK cross signals on 6h
    # Tenkan-sen (6h)
    period9_high_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6h = (period9_high_6h + period9_low_6h) / 2
    
    # Kijun-sen (6h)
    period26_high_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6h = (period26_high_6h + period26_low_6h) / 2
    
    # TK cross: Tenkan crosses above/below Kijun
    tk_cross_up = (tenkan_6h > kijun_6h) & (np.roll(tenkan_6h, 1) <= np.roll(kijun_6h, 1))
    tk_cross_down = (tenkan_6h < kijun_6h) & (np.roll(tenkan_6h, 1) >= np.roll(kijun_6h, 1))
    
    # Volume confirmation: 1.5x 20-bar average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Ichimoku calculations
    start_idx = max(52, 26)  # Senkou B needs 52, Kijun needs 26
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for TK cross signals in direction of 1d cloud with volume confirmation
            # Long: TK cross up in uptrend (cloud green)
            # Short: TK cross down in downtrend (cloud red)
            long_signal = tk_cross_up[i] and cloud_green[i] and volume_spike[i]
            short_signal = tk_cross_down[i] and cloud_red[i] and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when TK cross down (reverse signal) or price enters cloud (weakening trend)
            exit_signal = tk_cross_down[i] or (close[i] < senkou_a_aligned[i] and close[i] < senkou_b_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when TK cross up (reverse signal) or price enters cloud
            exit_signal = tk_cross_up[i] or (close[i] > senkou_a_aligned[i] and close[i] > senkou_b_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_1dCloudFilter"
timeframe = "6h"
leverage = 1.0