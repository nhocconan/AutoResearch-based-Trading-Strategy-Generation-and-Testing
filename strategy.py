#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend_Filter
Hypothesis: Use daily Ichimoku cloud direction as trend filter (bullish if price above cloud, bearish if below).
Enter on 6h Tenkan/Kijun cross in direction of daily trend, with volume confirmation.
Ichimoku provides dynamic support/resistance and trend identification that works in both bull/bear markets.
Target: 15-30 trades/year on 6h to minimize fee drag.
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
    
    # Get daily data for Ichimoku trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need enough for Ichimoku calculations
        return np.zeros(n)
    
    # Calculate Ichimoku components on daily data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tenkan_sen = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                      pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2).shift(26)
    
    # Align Ichimoku components to 6h timeframe (wait for daily close)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    # Determine if price is above or below the cloud on daily timeframe
    # Bullish trend: price above both Senkou spans
    # Bearish trend: price below both Senkou spans
    # Note: We use the close price from previous day to avoid look-ahead
    daily_close = df_1d['close'].values
    # Align daily close to 6h (previous day's close available at 6h open)
    daily_close_aligned = align_htf_to_ltf(prices, df_1d, daily_close)
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    cloud_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    # Trend determination: price relative to cloud
    bullish_trend = daily_close_aligned > cloud_top
    bearish_trend = daily_close_aligned < cloud_bottom
    
    # Calculate Tenkan/Kijun cross on 6h timeframe for entry signals
    high_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6h = (high_6h + low_6h) / 2
    
    high_6h_kj = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_6h_kj = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6h = (high_6h_kj + low_6h_kj) / 2
    
    # Cross signals: Tenkan crossing above/below Kijun
    tenkan_kijun_cross_up = (tenkan_6h > kijun_6h) & (np.roll(tenkan_6h, 1) <= np.roll(kijun_6h, 1))
    tenkan_kijun_cross_down = (tenkan_6h < kijun_6h) & (np.roll(tenkan_6h, 1) >= np.roll(kijun_6h, 1))
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for all calculations
    start_idx = max(52, 26, 20)  # Ichimoku needs 52, plus others
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(daily_close_aligned[i]) or np.isnan(tenkan_6h[i]) or 
            np.isnan(kijun_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check for trend and volume conditions
        is_bullish_trend = bullish_trend[i]
        is_bearish_trend = bearish_trend[i]
        has_vol_spike = vol_spike[i]
        tk_cross_up = tenkan_kijun_cross_up[i]
        tk_cross_down = tenkan_kijun_cross_down[i]
        
        if position == 0:
            # Long: bullish trend + Tenkan/Kijun cross up + volume spike
            if is_bullish_trend and tk_cross_up and has_vol_spike:
                signals[i] = size
                position = 1
            # Short: bearish trend + Tenkan/Kijun cross down + volume spike
            elif is_bearish_trend and tk_cross_down and has_vol_spike:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit conditions for long:
            # 1. Trend turns bearish (price below cloud)
            # 2. Tenkan/Kijun cross down (momentum shift)
            # 3. Loss of bullish structure (price below Kijun-sen)
            if (is_bearish_trend or tk_cross_down or close[i] < kijun_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit conditions for short:
            # 1. Trend turns bullish (price above cloud)
            # 2. Tenkan/Kijun cross up (momentum shift)
            # 3. Loss of bearish structure (price above Kijun-sen)
            if (is_bullish_trend or tk_cross_up or close[i] > kijun_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0