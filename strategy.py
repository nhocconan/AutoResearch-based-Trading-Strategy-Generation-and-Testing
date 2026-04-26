#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend_v1
Hypothesis: Trade Ichimoku Kumo Twist (Senkou Span A/B cross) on 6h with 1d EMA50 trend filter and volume confirmation (1.8x median). Kumo Twist signals trend acceleration. Only trade in direction of 1d EMA50 trend to reduce whipsaws. Target: 12-25 trades/year on 6h. Works in bull/bear by adapting to trend and using volume confirmation to filter false Kumo Twists.
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
    
    # Get 1d data for HTF trend and Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 26*2 for Ichimoku
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
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
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Kumo Twist: Senkou Span A crosses Senkou Span B
    # Bullish twist: Senkou Span A crosses above Senkou Span B
    # Bearish twist: Senkou Span A crosses below Senkou Span B
    senkou_span_a_shift = np.roll(senkou_span_a, 1)
    senkou_span_b_shift = np.roll(senkou_span_b, 1)
    senkou_span_a_shift[0] = np.nan
    senkou_span_b_shift[0] = np.nan
    
    bullish_twist = (senkou_span_a > senkou_span_b) & (senkou_span_a_shift <= senkou_span_b_shift)
    bearish_twist = (senkou_span_a < senkou_span_b) & (senkou_span_a_shift >= senkou_span_b_shift)
    
    # Align HTF indicators to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    bullish_twist_aligned = align_htf_to_ltf(prices, df_1d, bullish_twist.astype(float))
    bearish_twist_aligned = align_htf_to_ltf(prices, df_1d, bearish_twist.astype(float))
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Volume confirmation: 1.8x median volume (20-period) for signal
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of EMA(50) 1d, Ichimoku (52), volume median (20)
    start_idx = max(50, 52, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_median[i]) or
            np.isnan(bullish_twist_aligned[i]) or
            np.isnan(bearish_twist_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or
            np.isnan(senkou_span_b_aligned[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_50_1d_val = ema_50_1d_aligned[i]
        close_val = close[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        bullish_twist_val = bullish_twist_aligned[i] > 0.5
        bearish_twist_val = bearish_twist_aligned[i] > 0.5
        span_a = senkou_span_a_aligned[i]
        span_b = senkou_span_b_aligned[i]
        
        # Trend filter: price > EMA50 (uptrend) or < EMA50 (downtrend)
        uptrend = close_val > ema_50_1d_val
        downtrend = close_val < ema_50_1d_val
        
        if position == 0:
            # Long: bullish Kumo twist with volume spike, and uptrend
            long_signal = bullish_twist_val and \
                          (volume_val > 1.8 * vol_median_val) and \
                          uptrend
            
            # Short: bearish Kumo twist with volume spike, and downtrend
            short_signal = bearish_twist_val and \
                           (volume_val > 1.8 * vol_median_val) and \
                           downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long: exit when price closes below Kumo (Senkou Span A)
            signals[i] = 0.25
            if close_val < span_a:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short: exit when price closes above Kumo (Senkou Span B)
            signals[i] = -0.25
            if close_val > span_b:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrend_v1"
timeframe = "6h"
leverage = 1.0