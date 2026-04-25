#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1wTrend
Hypothesis: On 6h timeframe, trade Ichimoku TK cross signals filtered by 1w cloud (trend) and volume spike.
- Long: Tenkan crosses above Kijun + price above 1w cloud (bullish trend) + volume spike
- Short: Tenkan crosses below Kijun + price below 1w cloud (bearish trend) + volume spike
- Exit: Opposite TK cross or price crosses the opposite cloud edge (Tenkan-Kijun midpoint)
- Uses 1w cloud for major trend filter to avoid counter-trend whipsaws in ranging markets
- Volume spike ensures breakout validity
- Discrete position sizing (0.25) to limit fee drag
- Targets 12-25 trades/year (50-100 over 4 years)
Works in bull markets (trend-following with cloud support) and bear markets (trend-following with cloud resistance)
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
    
    # Get 1w data for cloud (Senkou Span A/B) and trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Ichimoku components on 1w (for cloud)
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 over 9 periods
    # Kijun-sen (Base Line): (highest high + lowest low)/2 over 26 periods
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2, plotted 26 periods ahead
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 over 52 periods, plotted 26 periods ahead
    
    # Calculate Tenkan and Kijun on 1w
    def calculate_ichimoku_components(high, low, close):
        n1 = len(high)
        tenkan = np.full(n1, np.nan)
        kijun = np.full(n1, np.nan)
        senkou_a = np.full(n1, np.nan)
        senkou_b = np.full(n1, np.nan)
        
        # Tenkan-sen (9-period)
        for i in range(8, n1):
            highest_high = np.max(high[i-8:i+1])
            lowest_low = np.min(low[i-8:i+1])
            tenkan[i] = (highest_high + lowest_low) / 2
        
        # Kijun-sen (26-period)
        for i in range(25, n1):
            highest_high = np.max(high[i-25:i+1])
            lowest_low = np.min(low[i-25:i+1])
            kijun[i] = (highest_high + lowest_low) / 2
        
        # Senkou Span B (52-period)
        for i in range(51, n1):
            highest_high = np.max(high[i-51:i+1])
            lowest_low = np.min(low[i-51:i+1])
            senkou_b[i] = (highest_high + lowest_low) / 2
        
        # Senkou Span A = (Tenkan + Kijun)/2
        # But note: Senkou spans are plotted 26 periods ahead
        # For alignment, we calculate the values and then shift when aligning
        for i in range(25, n1):  # Need both Tenkan and Kijun
            if not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
                senkou_a[i] = (tenkan[i] + kijun[i]) / 2
        
        return tenkan, kijun, senkou_a, senkou_b
    
    tenkan_1w, kijun_1w, senkou_a_1w, senkou_b_1w = calculate_ichimoku_components(high_1w, low_1w, close_1w)
    
    # Align 1w Ichimoku components to 6h (completed 1w bar only)
    tenkan_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_1w)
    kijun_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_1w)
    senkou_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_a_1w)
    senkou_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_1w)
    
    # Calculate TK cross on 6h timeframe for entry timing
    # Tenkan-sen (9-period) and Kijun-sen (26-period) on 6h
    tenkan_6h = np.full(n, np.nan)
    kijun_6h = np.full(n, np.nan)
    
    for i in range(8, n):
        highest_high = np.max(high[i-8:i+1])
        lowest_low = np.min(low[i-8:i+1])
        tenkan_6h[i] = (highest_high + lowest_low) / 2
    
    for i in range(25, n):
        highest_high = np.max(high[i-25:i+1])
        lowest_low = np.min(low[i-25:i+1])
        kijun_6h[i] = (highest_high + lowest_low) / 2
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 6h TK (26), 1w components (52 for Senkou B), volume MA (20)
    start_idx = max(26, 52, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or
            np.isnan(tenkan_1w_aligned[i]) or np.isnan(kijun_1w_aligned[i]) or
            np.isnan(senkou_a_1w_aligned[i]) or np.isnan(senkou_b_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine 1w cloud boundaries (Senkou Span A/B)
        # Cloud top = max(Senkou A, Senkou B), Cloud bottom = min(Senkou A, Senkou B)
        cloud_top = np.maximum(senkou_a_1w_aligned[i], senkou_b_1w_aligned[i])
        cloud_bottom = np.minimum(senkou_a_1w_aligned[i], senkou_b_1w_aligned[i])
        
        # TK cross signals on 6h
        tk_cross_up = tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1]
        tk_cross_down = tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1]
        
        if position == 0:
            # Long: TK cross up + price above cloud (bullish) + volume spike
            long_setup = tk_cross_up and (close[i] > cloud_top) and volume_spike[i]
            # Short: TK cross down + price below cloud (bearish) + volume spike
            short_setup = tk_cross_down and (close[i] < cloud_bottom) and volume_spike[i]
            
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
            # Exit: TK cross down OR price crosses below cloud bottom
            if tk_cross_down or (close[i] < cloud_bottom):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: TK cross up OR price crosses above cloud top
            if tk_cross_up or (close[i] > cloud_top):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1wTrend"
timeframe = "6h"
leverage = 1.0