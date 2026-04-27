#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_TK_Cross_1dTrend_Filter_v1
Hypothesis: Ichimoku TK cross with cloud filter on 6h, aligned with 1d trend (price vs EMA50), captures strong momentum moves while avoiding counter-trend whipsaws. Weekly trend filter (price vs 1w EMA50) avoids major counter-trend trades. Discrete sizing (0.25) balances return and fee drag. Target: 50-150 total trades over 4 years.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Get 1w data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Ichimoku components (6h)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1w EMA50 for weekly trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to primary timeframe (6h)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)  # tenkan is 6h, but we need to align it properly
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # For Ichimoku components that are already 6h, we need to handle alignment differently
    # Since tenkan, kijun, senkou_a, senkou_b are calculated from 6h data, we don't need HTF alignment for them
    # But we need to ensure they are properly indexed
    tenkan_6h = tenkan
    kijun_6h = kijun
    senkou_a_6h = senkou_a
    senkou_b_6h = senkou_b
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need Ichimoku (52), 1d EMA50 (50), 1w EMA50 (50)
    start_idx = max(52, 50, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        tenkan_val = tenkan_6h[i]
        kijun_val = kijun_6h[i]
        senkou_a_val = senkou_a_6h[i]
        senkou_b_val = senkou_b_6h[i]
        ema50_1d_val = ema50_1d_aligned[i]
        ema50_1w_val = ema50_1w_aligned[i]
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        if position == 0:
            # TK cross: Tenkan crosses above/below Kijun
            tk_cross_up = tenkan_val > kijun_val and tenkan_6h[i-1] <= kijun_6h[i-1]
            tk_cross_down = tenkan_val < kijun_val and tenkan_6h[i-1] >= kijun_6h[i-1]
            
            # Trend filters: price above/below EMAs
            uptrend_1d = close_val > ema50_1d_val
            uptrend_1w = close_val > ema50_1w_val
            downtrend_1d = close_val < ema50_1d_val
            downtrend_1w = close_val < ema50_1w_val
            
            # Long conditions: TK cross up + price above cloud + 1d and 1w uptrend
            if tk_cross_up and close_val > cloud_top and uptrend_1d and uptrend_1w:
                signals[i] = size
                position = 1
                entry_price = close_val
            # Short conditions: TK cross down + price below cloud + 1d and 1w downtrend
            elif tk_cross_down and close_val < cloud_bottom and downtrend_1d and downtrend_1w:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit conditions: TK cross down or price falls below cloud bottom
            tk_cross_down = tenkan_val < kijun_val and tenkan_6h[i-1] >= kijun_6h[i-1]
            if tk_cross_down or close_val < cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit conditions: TK cross up or price rises above cloud top
            tk_cross_up = tenkan_val > kijun_val and tenkan_6h[i-1] <= kijun_6h[i-1]
            if tk_cross_up or close_val > cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_1dTrend_Filter_v1"
timeframe = "6h"
leverage = 1.0