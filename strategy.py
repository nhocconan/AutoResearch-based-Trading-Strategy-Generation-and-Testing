#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_VolumeSpike_Trend
Hypothesis: Trade 6h Ichimoku Kumo twist (Senkou Span A/B cross) with 1w trend filter (price >/<
1w EMA50) and volume confirmation (>2.0x 24-bar MA). Kumo twist indicates trend acceleration,
working in both bull (long twists) and bear (short twists). Target: 12-25 trades/year
(50-100 over 4 years) to minimize fee drag. Discrete sizing 0.25.
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Ichimoku components (using 6h data)
    # Conversion Line (Tenkan-sen): (9-period high + low)/2
    period_9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period_9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period_9_high + period_9_low) / 2
    
    # Base Line (Kijun-sen): (26-period high + low)/2
    period_26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period_26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period_26_high + period_26_low) / 2
    
    # Leading Span A (Senkou Span A): (Tenkan-sen + Kijun-sen)/2 plotted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Leading Span B (Senkou Span B): (52-period high + low)/2 plotted 26 periods ahead
    period_52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period_52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period_52_high + period_52_low) / 2)
    
    # Align Ichimoku components to current time (shift back 26 periods for look-ahead avoidance)
    # Since Senkou spans are plotted 26 periods ahead, we need to compare current price
    # with Senkou values from 26 periods ago
    senkou_span_a_lagged = np.roll(senkou_span_a, 26)
    senkou_span_b_lagged = np.roll(senkou_span_b, 26)
    # First 26 values are invalid due to roll
    senkou_span_a_lagged[:26] = np.nan
    senkou_span_b_lagged[:26] = np.nan
    
    # Kumo twist: Senkou Span A crosses above/below Senkou Span B
    # Twist up (bullish): Senkou A crosses above Senkou B
    twist_up = (senkou_span_a_lagged > senkou_span_b_lagged) & (np.roll(senkou_span_a_lagged, 1) <= np.roll(senkou_span_b_lagged, 1))
    # Twist down (bearish): Senkou A crosses below Senkou B
    twist_down = (senkou_span_a_lagged < senkou_span_b_lagged) & (np.roll(senkou_span_a_lagged, 1) >= np.roll(senkou_span_b_lagged, 1))
    
    # Volume confirmation: current volume > 2.0x 24-period average
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 1w EMA50 (50), Ichimoku (52), volume MA (24)
    start_idx = max(50, 52, 24) + 26  # +26 for Senkou lag
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(twist_up[i]) or np.isnan(twist_down[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Kumo twist up AND price > 1w EMA50 (bullish trend) AND volume confirm
            long_setup = twist_up[i] and \
                         (close[i] > ema_50_1w_aligned[i]) and \
                         volume_confirm[i]
            # Short: Kumo twist down AND price < 1w EMA50 (bearish trend) AND volume confirm
            short_setup = twist_down[i] and \
                          (close[i] < ema_50_1w_aligned[i]) and \
                          volume_confirm[i]
            
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
            # Exit: Kumo twist down OR price < 1w EMA50
            if twist_down[i] or (close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: Kumo twist up OR price > 1w EMA50
            if twist_up[i] or (close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_VolumeSpike_Trend"
timeframe = "6h"
leverage = 1.0