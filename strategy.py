#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_TK_Cross_1dTrend
Hypothesis: Use 6h timeframe with Ichimoku TK crossover, confirmed by 1d cloud trend.
Long when: Tenkan crosses above Kijun + price above cloud + 1d cloud bullish.
Short when: Tenkan crosses below Kijun + price below cloud + 1d cloud bearish.
Exit when: TK cross reverses or price exits cloud.
Uses discrete 0.25 position size to limit fee drag. Designed for BTC/ETH:
- Works in trending markets via cloud filter
- TK cross provides timely entries
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
    
    # Cloud (Kumo): Senkou Span A and B
    # Upper cloud: max(Senkou A, Senkou B)
    # Lower cloud: min(Senkou A, Senkou B)
    upper_cloud = np.maximum(senkou_a, senkou_b)
    lower_cloud = np.minimum(senkou_a, senkou_b)
    
    # TK Cross signals
    tk_cross_up = (tenkan > kijun) & (tenkan <= kijun)  # Tenkan crossed above Kijun
    tk_cross_down = (tenkan < kijun) & (tenkan >= kijun)  # Tenkan crossed below Kijun
    
    # 1d HTF for cloud trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate 1d Ichimoku cloud
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    d_period9_high = pd.Series(d_high).rolling(window=9, min_periods=9).max().values
    d_period9_low = pd.Series(d_low).rolling(window=9, min_periods=9).min().values
    d_tenkan = (d_period9_high + d_period9_low) / 2
    
    d_period26_high = pd.Series(d_high).rolling(window=26, min_periods=26).max().values
    d_period26_low = pd.Series(d_low).rolling(window=26, min_periods=26).min().values
    d_kijun = (d_period26_high + d_period26_low) / 2
    
    d_senkou_a = (d_tenkan + d_kijun) / 2
    d_period52_high = pd.Series(d_high).rolling(window=52, min_periods=52).max().values
    d_period52_low = pd.Series(d_low).rolling(window=52, min_periods=52).min().values
    d_senkou_b = (d_period52_high + d_period52_low) / 2
    
    d_upper_cloud = np.maximum(d_senkou_a, d_senkou_b)
    d_lower_cloud = np.minimum(d_senkou_a, d_senkou_b)
    
    # 1d cloud trend: bullish if close above upper cloud, bearish if below lower cloud
    d_1d_cloud_bullish = d_close > d_upper_cloud
    d_1d_cloud_bearish = d_close < d_lower_cloud
    
    # Align 1d cloud trend to 6h
    d_1d_cloud_bullish_aligned = align_htf_to_ltf(prices, df_1d, d_1d_cloud_bullish.astype(float))
    d_1d_cloud_bearish_aligned = align_htf_to_ltf(prices, df_1d, d_1d_cloud_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 52 for Senkou B
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(upper_cloud[i]) or
            np.isnan(lower_cloud[i]) or np.isnan(d_1d_cloud_bullish_aligned[i]) or
            np.isnan(d_1d_cloud_bearish_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # Fixed position size
        
        if position == 0:
            # Flat - look for TK cross with cloud and 1d trend confirmation
            # Long: TK cross up + price above cloud + 1d cloud bullish
            long_entry = tk_cross_up[i] and (close_val > upper_cloud[i]) and d_1d_cloud_bullish_aligned[i]
            # Short: TK cross down + price below cloud + 1d cloud bearish
            short_entry = tk_cross_down[i] and (close_val < lower_cloud[i]) and d_1d_cloud_bearish_aligned[i]
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when TK cross down or price exits cloud below
            if tk_cross_down[i] or (close_val < lower_cloud[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when TK cross up or price exits cloud above
            if tk_cross_up[i] or (close_val > upper_cloud[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_1dTrend"
timeframe = "6h"
leverage = 1.0