#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_v1
Hypothesis: Trade Ichimoku Tenkan-Kijun crosses on 6h timeframe with daily cloud filter. 
Long when TK crosses bullish AND price above daily cloud (Senkou Span A/B). 
Short when TK crosses bearish AND price below daily cloud. 
Exit on opposite TK cross or cloud color change. 
Position size: 0.25 to limit drawdown. 
Target: 50-150 total trades over 4 years = 12-37/year. 
Works in bull (TK bullish + price above cloud) and bear (TK bearish + price below cloud) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need sufficient data for Ichimoku (26*2)
        return np.zeros(n)
    
    # Calculate 6h Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2
    
    # Calculate daily Ichimoku cloud components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen 1d (9-period)
    high_tenkan_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_tenkan_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (high_tenkan_1d + low_tenkan_1d) / 2
    
    # Kijun-sen 1d (26-period)
    high_kijun_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_kijun_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (high_kijun_1d + low_kijun_1d) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2 plotted 26 periods ahead
    senkou_span_a_1d = ((tenkan_1d + kijun_1d) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2 plotted 26 periods ahead
    high_senkou_b_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_senkou_b_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b_1d = (high_senkou_b_1d + low_senkou_b_1d) / 2
    
    # Align 6h Ichimoku components
    tenkan_aligned = align_htf_to_ltf(prices, prices, tenkan)  # Same timeframe, no alignment needed but using helper for consistency
    kijun_aligned = align_htf_to_ltf(prices, prices, kijun)
    
    # Align daily cloud components to 6h timeframe
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d)
    
    # Determine cloud top and bottom (Senkou Span A and B)
    cloud_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    cloud_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    # Cloud color: green when Senkou Span A > Senkou Span B (bullish), red when < (bearish)
    cloud_bullish = senkou_span_a_aligned > senkou_span_b_aligned
    cloud_bearish = senkou_span_a_aligned < senkou_span_b_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Kijun (26) and daily components (52)
    start_idx = max(26, 52)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or
            np.isnan(cloud_top[i]) or
            np.isnan(cloud_bottom[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine TK cross
        tk_bullish_cross = (tenkan_aligned[i] > kijun_aligned[i]) and (tenkan_aligned[i-1] <= kijun_aligned[i-1])
        tk_bearish_cross = (tenkan_aligned[i] < kijun_aligned[i]) and (tenkan_aligned[i-1] >= kijun_aligned[i-1])
        
        # Price relative to cloud
        price_above_cloud = close[i] > cloud_top[i]
        price_below_cloud = close[i] < cloud_bottom[i]
        
        if position == 0:
            # Long setup: bullish TK cross + price above cloud
            long_setup = tk_bullish_cross and price_above_cloud
            
            # Short setup: bearish TK cross + price below cloud
            short_setup = tk_bearish_cross and price_below_cloud
            
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
            # Exit: bearish TK cross OR price falls below cloud
            if tk_bearish_cross or (close[i] < cloud_top[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: bullish TK cross OR price rises above cloud
            if tk_bullish_cross or (close[i] > cloud_bottom[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_v1"
timeframe = "6h"
leverage = 1.0