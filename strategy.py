#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1wTrend_v1
Hypothesis: Ichimoku cloud twist (Senkou Span A/B cross) on 6h with 1w trend filter (price >/ < 1w EMA50) and volume confirmation (1.5x 20-period median) for breakout continuation. 
Works in bull/bear via 1w trend filter: only long in 1w uptrend, short in 1w downtrend. 
Cloud twist signals momentum shift; volume confirms institutional interest. 
Target: 60-120 trades over 4 years (15-30/year) to avoid fee drag.
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
    
    # Get 1w data for HTF trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Ichimoku components on 6h data
    # Conversion line (Tenkan-sen): (9-period high + low)/2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Base line (Kijun-sen): (26-period high + low)/2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Leading Span A (Senkou Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Leading Span B (Senkou Span B): (52-period high + low)/2 shifted 26 periods ahead
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_52 + low_52) / 2)
    
    # Align HTF indicators to 6h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: 1.5x median volume (20-period) for signal
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of Ichimoku (52), 1w EMA50, volume median (20)
    start_idx = max(52, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(vol_median[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_50_1w_val = ema_50_1w_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        
        # Ichimoku values at current bar
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        senkou_a_val = senkou_a[i]
        senkou_b_val = senkou_b[i]
        
        # Cloud twist: Senkou Span A crossing above/below Senkou Span B
        # Use previous bar to detect cross
        if i >= 1:
            senkou_a_prev = senkou_a[i-1]
            senkou_b_prev = senkou_b[i-1]
            twist_up = (senkou_a_prev <= senkou_b_prev) and (senkou_a_val > senkou_b_val)
            twist_down = (senkou_a_prev >= senkou_b_prev) and (senkou_a_val < senkou_b_val)
        else:
            twist_up = False
            twist_down = False
        
        # Trend filter: price > 1w EMA50 (uptrend) or < 1w EMA50 (downtrend)
        uptrend = close_val > ema_50_1w_val
        downtrend = close_val < ema_50_1w_val
        
        if position == 0:
            # Long: cloud twist up + price above cloud + uptrend + volume
            long_signal = twist_up and \
                          (close_val > max(senkou_a_val, senkou_b_val)) and \
                          uptrend and \
                          (volume_val > 1.5 * vol_median_val)
            
            # Short: cloud twist down + price below cloud + downtrend + volume
            short_signal = twist_down and \
                           (close_val < min(senkou_a_val, senkou_b_val)) and \
                           downtrend and \
                           (volume_val > 1.5 * vol_median_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long: exit on twist down or price closes below cloud
            signals[i] = 0.25
            if twist_down or close_val < min(senkou_a_val, senkou_b_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short: exit on twist up or price closes above cloud
            signals[i] = -0.25
            if twist_up or close_val > max(senkou_a_val, senkou_b_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1wTrend_v1"
timeframe = "6h"
leverage = 1.0