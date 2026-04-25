#!/usr/bin/env python3
"""
6h_Ichimoku_Trend_CloudBreakout_v1
Hypothesis: Use Ichimoku cloud from 1d HTF for trend direction and 6h price for breakout entries.
Long when 6h price breaks above Kumo cloud top in 1d uptrend (price > Senkou Span B + TK cross bullish).
Short when 6h price breaks below Kumo cloud bottom in 1d downtrend (price < Senkou Span A + TK cross bearish).
Exit when price re-enters the cloud or TK cross reverses.
Position size: 0.25 to balance profit and fee drag.
Target: 12-30 trades/year on 6h (50-120 total over 4 years) to avoid fee drag.
Ichimoku works in both bull and bear markets by adapting to trend via cloud and TK cross.
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
    
    # Get 1d data for Ichimoku HTF
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    displacement = 26
    
    # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past 9 periods
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).max().values
    
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).min().values
    
    tenkan_sen = (rolling_max(high_1d, tenkan_period) + rolling_min(low_1d, tenkan_period)) / 2.0
    # Calculate Kijun-sen (Base Line): (highest high + lowest low)/2 for past 26 periods
    kijun_sen = (rolling_max(high_1d, kijun_period) + rolling_min(low_1d, kijun_period)) / 2.0
    # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2.0
    # Calculate Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past 52 periods
    senkou_span_b = (rolling_max(high_1d, senkou_span_b_period) + rolling_min(low_1d, senkou_span_b_period)) / 2.0
    
    # Align Ichimoku lines to 6h timeframe (displaced forward by 26 periods)
    # For Senkou Span A/B, we need to shift them forward by displacement periods
    # align_htf_to_ltf already handles completed-bar timing, so we align the values
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # TK cross signals (bullish when Tenkan > Kijun)
    tk_bullish = tenkan_aligned > kijun_aligned
    tk_bearish = tenkan_aligned < kijun_aligned
    
    # Cloud top and bottom (in uptrend: Senkou Span A > Senkou Span B)
    cloud_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    cloud_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    # Volume confirmation: 6h volume > 1.5 * 20-period average
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Ichimoku (52) and volume MA (20)
    start_idx = max(52 + displacement, 20)  # 52+26=78 for Senkou Span, plus displacement handled in align
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long setup: price breaks above cloud top + TK bullish + volume spike
            long_setup = (close[i] > cloud_top[i]) and tk_bullish[i] and volume_spike[i]
            
            # Short setup: price breaks below cloud bottom + TK bearish + volume spike
            short_setup = (close[i] < cloud_bottom[i]) and tk_bearish[i] and volume_spike[i]
            
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
            # Exit: price re-enters cloud (below cloud top) OR TK cross turns bearish
            if (close[i] < cloud_top[i]) or (not tk_bullish[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters cloud (above cloud bottom) OR TK cross turns bullish
            if (close[i] > cloud_bottom[i]) or (tk_bullish[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Trend_CloudBreakout_v1"
timeframe = "6h"
leverage = 1.0