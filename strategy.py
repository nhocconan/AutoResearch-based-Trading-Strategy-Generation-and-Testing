#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_v1
Hypothesis: 6h Ichimoku Tenkan-Kijun cross with 1d cloud filter and volume confirmation.
- Uses 6h timeframe to capture medium-term swings with lower frequency than 1h/4h
- Ichimoku TK cross provides momentum signals (bullish when Tenkan > Kijun)
- 1d Ichimoku cloud acts as trend filter: only take longs when price above cloud, shorts when below cloud
- Volume spike (1.8x 20-period average) confirms institutional participation
- Designed for 15-35 trades/year (60-140 total over 4 years) to minimize fee drag
- Works in bull/bear markets by using 1d cloud as primary trend filter and TK cross for timing
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_9 = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_1d = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_26 = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_1d = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    max_high_52 = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b_1d = (max_high_52 + min_low_52) / 2
    
    # Align 1d Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Calculate 6h Ichimoku components for TK cross
    max_high_9_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    min_low_9_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6h = (max_high_9_6h + min_low_9_6h) / 2
    
    max_high_26_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    min_low_26_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6h = (max_high_26_6h + min_low_26_6h) / 2
    
    # Volume spike: volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 52 for Senkou B, 26 for Kijun, 20 for volume MA)
    start_idx = max(52, 26, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Determine cloud boundaries (Senkou Span A/B)
        upper_cloud = max(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        lower_cloud = min(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        # TK cross conditions
        tk_cross_bullish = tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1]
        tk_cross_bearish = tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1]
        
        # Price relative to cloud
        price_above_cloud = close[i] > upper_cloud
        price_below_cloud = close[i] < lower_cloud
        
        if position == 0:
            # Long: bullish TK cross AND price above cloud AND volume spike
            if tk_cross_bullish and price_above_cloud and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish TK cross AND price below cloud AND volume spike
            elif tk_cross_bearish and price_below_cloud and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: bearish TK cross OR price drops below cloud
            if tk_cross_bearish or not price_above_cloud:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: bullish TK cross OR price rises above cloud
            if tk_cross_bullish or not price_below_cloud:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_v1"
timeframe = "6h"
leverage = 1.0