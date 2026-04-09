#!/usr/bin/env python3
# 6h_1w_ichimoku_cloud_v1
# Hypothesis: 6h strategy using weekly Ichimoku cloud for trend direction and 6h TK cross for entry timing.
# Long: Price above weekly Kumo (cloud), Tenkan-sen crosses above Kijun-sen on 6h.
# Short: Price below weekly Kumo (cloud), Tenkan-sen crosses below Kijun-sen on 6h.
# Exit: Opposite TK cross or price crosses Kumo midpoint (Senkou Span B).
# Uses weekly Ichimoku for structure (works in bull/bear via cloud filter), 6h for precise entries.
# Target: 12-37 trades/year (50-150 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_ichimoku_cloud_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data for Ichimoku (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly Ichimoku components
    high_1w = pd.Series(df_1w['high'].values)
    low_1w = pd.Series(df_1w['low'].values)
    close_1w = pd.Series(df_1w['close'].values)
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = high_1w.rolling(window=9, min_periods=9).max()
    period9_low = low_1w.rolling(window=9, min_periods=9).min()
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = high_1w.rolling(window=26, min_periods=26).max()
    period26_low = low_1w.rolling(window=26, min_periods=26).min()
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = high_1w.rolling(window=52, min_periods=52).max()
    period52_low = low_1w.rolling(window=52, min_periods=52).min()
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Align HTF Ichimoku components to 6h timeframe (wait for completed 1w bar)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b.values)
    
    # Calculate 6h Tenkan-sen and Kijun-sen for TK cross
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    period9_high_6h = high_s.rolling(window=9, min_periods=9).max()
    period9_low_6h = low_s.rolling(window=9, min_periods=9).min()
    tenkan_sen_6h = (period9_high_6h + period9_low_6h) / 2
    
    period26_high_6h = high_s.rolling(window=26, min_periods=26).max()
    period26_low_6h = low_s.rolling(window=26, min_periods=26).min()
    kijun_sen_6h = (period26_high_6h + period26_low_6h) / 2
    
    # Kumo midpoint (Senkou Span B) for exit
    kumo_midpoint_aligned = (senkou_span_a_aligned + senkou_span_b_aligned) / 2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i]) or
            np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i])):
            signals[i] = 0.0
            continue
        
        # Determine if price is above/below weekly Kumo (cloud)
        # Cloud top = max(Senkou Span A, Senkou Span B)
        # Cloud bottom = min(Senkou Span A, Senkou Span B)
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        price_above_kumo = close[i] > cloud_top
        price_below_kumo = close[i] < cloud_bottom
        
        # TK cross on 6h
        tk_cross_bullish = tenkan_sen_6h[i] > kijun_sen_6h[i] and tenkan_sen_6h[i-1] <= kijun_sen_6h[i-1]
        tk_cross_bearish = tenkan_sen_6h[i] < kijun_sen_6h[i] and tenkan_sen_6h[i-1] >= kijun_sen_6h[i-1]
        
        if position == 1:  # Long position
            # Exit: TK cross bearish or price crosses below Kumo midpoint
            if tk_cross_bearish or close[i] < kumo_midpoint_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TK cross bullish or price crosses above Kumo midpoint
            if tk_cross_bullish or close[i] > kumo_midpoint_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long: Price above weekly Kumo and bullish TK cross
            if price_above_kumo and tk_cross_bullish:
                position = 1
                signals[i] = 0.25
            # Short: Price below weekly Kumo and bearish TK cross
            elif price_below_kumo and tk_cross_bearish:
                position = -1
                signals[i] = -0.25
    
    return signals