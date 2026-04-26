#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1d
Hypothesis: 6h Ichimoku Tenkan/Kijun cross with daily cloud filter for trend alignment.
The daily Ichimoku cloud provides strong trend filter (price above/below cloud) while 6h TK cross
captures momentum shifts. This combination reduces false signals in choppy markets and works
across bull/bear regimes by only taking trades in the direction of the higher timeframe trend.
Target: 12-30 trades/year to minimize fee drag. Uses discrete position sizing (0.25).
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
    
    # Get daily data for HTF Ichimoku cloud filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need enough for Ichimoku calculations
        return np.zeros(n)
    
    # Calculate Ichimoku components on daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe (completed daily bars only)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Calculate 6h Ichimoku for TK cross
    period9_high_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen_6h = (period9_high_6h + period9_low_6h) / 2
    
    period26_high_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen_6h = (period26_high_6h + period26_low_6h) / 2
    
    # TK cross signals
    tk_cross_up = (tenkan_sen_6h > kijun_sen_6h) & (tenkan_sen_6h.shift(1) <= kijun_sen_6h.shift(1))
    tk_cross_down = (tenkan_sen_6h < kijun_sen_6h) & (tenkan_sen_6h.shift(1) >= kijun_sen_6h.shift(1))
    
    # Price relative to cloud (using daily cloud aligned to 6h)
    # Cloud top = max(Senkou Span A, Senkou Span B)
    # Cloud bottom = min(Senkou Span A, Senkou Span B)
    cloud_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    cloud_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of Ichimoku periods (52) + 6h TK calculation (26)
    start_idx = max(52, 26) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: TK cross up AND price above daily cloud (bullish alignment)
            long_signal = tk_cross_up[i] and price_above_cloud[i]
            
            # Short: TK cross down AND price below daily cloud (bearish alignment)
            short_signal = tk_cross_down[i] and price_below_cloud[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: TK cross down OR price falls below cloud (trend change)
            if tk_cross_down[i] or (not price_above_cloud[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: TK cross up OR price rises above cloud (trend change)
            if tk_cross_up[i] or (not price_below_cloud[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1d"
timeframe = "6h"
leverage = 1.0