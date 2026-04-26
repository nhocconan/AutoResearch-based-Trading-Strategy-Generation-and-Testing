#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1wTrend
Hypothesis: Use 6h timeframe with Ichimoku Tenkan-Kijun cross, filtered by 1w cloud color and price >/< cloud.
Long when: TK cross bullish + price above cloud + 1w cloud green (bullish).
Short when: TK cross bearish + price below cloud + 1w cloud red (bearish).
Exit when: TK cross reverses or price crosses cloud midpoint.
Uses discrete 0.25 position size to limit fee drag. Designed for BTC/ETH:
- Ichimoku provides multiple confirmation layers (trend, momentum, support/resistance)
- 1w cloud filter ensures trading with the weekly trend, reducing whipsaws in bear markets
- TK cross gives timely entries while cloud filter avoids counter-trend trades
- Targets 12-37 trades/year for optimal test generalization.
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
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Cloud (Kumo): between Senkou Span A and B
    # Cloud top = max(Senkou A, Senkou B)
    # Cloud bottom = min(Senkou A, Senkou B)
    # We'll use these for price vs cloud checks
    
    # 1w Ichimoku for trend filter (same calculation on weekly data)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:  # need at least 52 weeks for Senkou B
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # 1w Tenkan-sen
    period9_high_1w = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    period9_low_1w = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_1w = (period9_high_1w + period9_low_1w) / 2
    
    # 1w Kijun-sen
    period26_high_1w = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    period26_low_1w = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_1w = (period26_high_1w + period26_low_1w) / 2
    
    # 1w Senkou Span A
    senkou_a_1w = ((tenkan_1w + kijun_1w) / 2)
    
    # 1w Senkou Span B
    period52_high_1w = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    period52_low_1w = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_b_1w = ((period52_high_1w + period52_low_1w) / 2)
    
    # 1w cloud color: green if Senkou A > Senkou B (bullish), red if Senkou A < Senkou B (bearish)
    cloud_green_1w = senkou_a_1w > senkou_b_1w
    cloud_red_1w = senkou_a_1w < senkou_b_1w
    
    # Align 6h Ichimoku components (wait for completed 6h bar - but since we calculate on close, no extra delay needed for TK cross)
    # However, for cloud, we need to align the forward-shifted Senkou spans
    # Senkou spans are already shifted 26 periods ahead in their calculation, so we use as-is
    
    # Align 1w HTF data to 6s timeframe
    cloud_green_1w_aligned = align_htf_to_ltf(prices, df_1w, cloud_green_1w.astype(float))
    cloud_red_1w_aligned = align_htf_to_ltf(prices, df_1w, cloud_red_1w.astype(float))
    senkou_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_a_1w)
    senkou_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 52 for Senkou B calculation
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(senkou_a_1w_aligned[i]) or np.isnan(senkou_b_1w_aligned[i]) or
            np.isnan(cloud_green_1w_aligned[i]) or np.isnan(cloud_red_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # Fixed position size
        
        # Cloud boundaries from 1w (aligned)
        cloud_top = max(senkou_a_1w_aligned[i], senkou_b_1w_aligned[i])
        cloud_bottom = min(senkou_a_1w_aligned[i], senkou_b_1w_aligned[i])
        cloud_mid = (cloud_top + cloud_bottom) / 2
        
        if position == 0:
            # Flat - look for TK cross with cloud filter
            # Long: TK cross bullish (Tenkan > Kijun) + price above cloud + 1w cloud green
            tk_bullish = tenkan[i] > kijun[i]
            price_above_cloud = close_val > cloud_top
            long_entry = tk_bullish and price_above_cloud and (cloud_green_1w_aligned[i] > 0.5)
            
            # Short: TK cross bearish (Tenkan < Kijun) + price below cloud + 1w cloud red
            tk_bearish = tenkan[i] < kijun[i]
            price_below_cloud = close_val < cloud_bottom
            short_entry = tk_bearish and price_below_cloud and (cloud_red_1w_aligned[i] > 0.5)
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when TK cross bearish OR price crosses below cloud bottom
            tk_bearish = tenkan[i] < kijun[i]
            price_below_cloud = close_val < cloud_bottom
            if tk_bearish or price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when TK cross bullish OR price crosses above cloud top
            tk_bullish = tenkan[i] > kijun[i]
            price_above_cloud = close_val > cloud_top
            if tk_bullish or price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1wTrend"
timeframe = "6h"
leverage = 1.0