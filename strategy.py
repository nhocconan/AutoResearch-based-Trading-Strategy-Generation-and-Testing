#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Twist_12hTrend_v1
Hypothesis: 6h Ichimoku TK cross with 12h cloud filter and volume confirmation.
In bull markets: price above cloud + TK cross up = momentum continuation.
In bear markets: price below cloud + TK cross down = trend acceleration.
The 12h cloud acts as a higher timeframe trend filter to avoid counter-trend whipsaws.
Volume confirmation ensures breakout conviction. Discrete sizing (0.25) limits fee drag.
Target: 50-150 total trades over 4 years (12-37/year) by requiring TK cross, cloud alignment, and volume spike.
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
    
    # Load 12h data ONCE before loop for HTF Ichimoku cloud
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Ichimoku components
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_12h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_12h).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_12h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_12h).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_12h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_12h).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2.0
    
    # Align HTF Ichimoku components to 6h timeframe (completed 12h bars only)
    tenkan_aligned = align_htf_to_ltf(prices, df_12h, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_12h, kijun_sen)
    span_a_aligned = align_htf_to_ltf(prices, df_12h, senkou_span_a)
    span_b_aligned = align_htf_to_ltf(prices, df_12h, senkou_span_b)
    
    # 6h volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA, 52 for Senkou B)
    start_idx = max(20, 52)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or
            np.isnan(span_a_aligned[i]) or
            np.isnan(span_b_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition
        volume_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Ichimoku TK cross conditions
        tk_cross_up = tenkan_aligned[i] > kijun_aligned[i]  # Tenkan crosses above Kijun
        tk_cross_down = tenkan_aligned[i] < kijun_aligned[i]  # Tenkan crosses below Kijun
        
        # Cloud conditions: price above/both spans (bullish) or below/both spans (bearish)
        cloud_top = np.maximum(span_a_aligned[i], span_b_aligned[i])
        cloud_bottom = np.minimum(span_a_aligned[i], span_b_aligned[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        if tk_cross_up and price_above_cloud and volume_spike:
            # Long signal: TK cross up + price above cloud + volume spike
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        elif tk_cross_down and price_below_cloud and volume_spike:
            # Short signal: TK cross down + price below cloud + volume spike
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Twist_12hTrend_v1"
timeframe = "6h"
leverage = 1.0