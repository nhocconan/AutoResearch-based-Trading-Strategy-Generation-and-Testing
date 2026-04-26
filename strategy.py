#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_WeeklyTrend_Filter
Hypothesis: 6h Ichimoku TK cross with weekly cloud filter and volume confirmation. 
Long when TK cross above cloud in weekly uptrend, short when TK cross below cloud in weekly downtrend. 
Uses weekly Ichimoku for trend direction (avoids whipsaws in bear markets) and 6h for precise entry timing. 
Volume confirmation (>1.5x average) ensures conviction. Discrete sizing 0.25 targets ~20 trades/year (80 total over 4 years) to minimize fee drag.
Designed for both bull and bear markets: weekly trend filter adapts to higher timeframe momentum.
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
    
    # Get weekly data for Ichimoku trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:  # Need enough for weekly Ichimoku
        return np.zeros(n)
    
    # Weekly Ichimoku components
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_10 = pd.Series(high_1w).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_10 = pd.Series(low_1w).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_1w = (max_high_10 + min_low_10) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_26 = pd.Series(high_1w).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_1w).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_1w = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_a_1w = (tenkan_1w + kijun_1w) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period_senkou_b = 52
    max_high_52 = pd.Series(high_1w).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low_1w).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b_1w = (max_high_52 + min_low_52) / 2
    
    # Align weekly Ichimoku to 6h
    tenkan_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_1w)
    kijun_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_1w)
    senkou_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_a_1w)
    senkou_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_1w)
    
    # Weekly trend: price above/both Senkou spans
    weekly_uptrend = (senkou_a_1w_aligned > senkou_b_1w_aligned) & (close_1w[-1] > senkou_a_1w_aligned) if len(close_1w) > 0 else False
    weekly_downtrend = (senkou_a_1w_aligned < senkou_b_1w_aligned) & (close_1w[-1] < senkou_b_1w_aligned) if len(close_1w) > 0 else False
    
    # 6h Ichimoku for entry signals
    period_tenkan_6h = 9
    period_kijun_6h = 26
    
    max_high_10_6h = pd.Series(high).rolling(window=period_tenkan_6h, min_periods=period_tenkan_6h).max().values
    min_low_10_6h = pd.Series(low).rolling(window=period_tenkan_6h, min_periods=period_tenkan_6h).min().values
    tenkan_6h = (max_high_10_6h + min_low_10_6h) / 2
    
    max_high_26_6h = pd.Series(high).rolling(window=period_kijun_6h, min_periods=period_kijun_6h).max().values
    min_low_26_6h = pd.Series(low).rolling(window=period_kijun_6h, min_periods=period_kijun_6h).min().values
    kijun_6h = (max_high_26_6h + min_low_26_6h) / 2
    
    # TK Cross signals
    tk_cross_above = (tenkan_6h > kijun_6h) & (tenkan_6h[:-1] <= kijun_6h[:-1])  # Bullish cross
    tk_cross_below = (tenkan_6h < kijun_6h) & (tenkan_6h[:-1] >= kijun_6h[:-1])  # Bearish cross
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of weekly Ichimoku (52), 6h TK (26), volume MA (20)
    start_idx = max(52, 26, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_1w_aligned[i]) or 
            np.isnan(kijun_1w_aligned[i]) or 
            np.isnan(senkou_a_1w_aligned[i]) or 
            np.isnan(senkou_b_1w_aligned[i]) or
            np.isnan(tenkan_6h[i]) or
            np.isnan(kijun_6h[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        tenkan_1w_val = tenkan_1w_aligned[i]
        kijun_1w_val = kijun_1w_aligned[i]
        senkou_a_1w_val = senkou_a_1w_aligned[i]
        senkou_b_1w_val = senkou_b_1w_aligned[i]
        tenkan_6h_val = tenkan_6h[i]
        kijun_6h_val = kijun_6h[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        
        # Weekly trend determination
        cloud_top = max(senkou_a_1w_val, senkou_b_1w_val)
        cloud_bottom = min(senkou_a_1w_val, senkou_b_1w_val)
        weekly_uptrend = senkou_a_1w_val > senkou_b_1w_val
        weekly_downtrend = senkou_a_1w_val < senkou_b_1w_val
        
        # Volume confirmation
        volume_confirmed = vol_val > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: TK cross above in weekly uptrend + volume confirmation
            long_signal = tk_cross_above[i] and weekly_uptrend and volume_confirmed
            # Short: TK cross below in weekly downtrend + volume confirmation
            short_signal = tk_cross_below[i] and weekly_downtrend and volume_confirmed
            
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
            # Exit: TK cross below OR weekly trend changes to downtrend
            if tk_cross_below[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: TK cross above OR weekly trend changes to uptrend
            if tk_cross_above[i] or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_WeeklyTrend_Filter"
timeframe = "6h"
leverage = 1.0