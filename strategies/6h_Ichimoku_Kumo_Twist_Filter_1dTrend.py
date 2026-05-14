#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_Filter_1dTrend
Hypothesis: Ichimoku TK cross with Kumo twist confirmation on 6h, filtered by 1d trend (price vs Kumo).
Long when TK crosses above AND price above Kumo (bullish twist). Short when TK crosses below AND price below Kumo (bearish twist).
Uses discrete sizing (0.25) to minimize fee drag. Target: 50-150 trades over 4 years.
Works in bull/bear via Kumo filter (trend proxy) and TK cross for momentum.
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
    
    # Ichimoku components (9, 26, 52 periods)
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
    
    # Load 1d data for HTF trend filter (Kumo twist)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d Senkou Span A and B for Kumo twist
    period9_high_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (period9_high_1d + period9_low_1d) / 2
    
    period26_high_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (period26_high_1d + period26_low_1d) / 2
    
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    period52_high_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (period52_high_1d + period52_low_1d) / 2
    
    # Align 1d Ichimoku components to 6h
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 52-period for Senkou B)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Kumo twist: Senkou A crossing Senkou B on 1d
        # Bullish twist: Senkou A > Senkou B
        # Bearish twist: Senkou A < Senkou B
        kumo_bullish = senkou_a_1d_aligned[i] > senkou_b_1d_aligned[i]
        kumo_bearish = senkou_a_1d_aligned[i] < senkou_b_1d_aligned[i]
        
        # TK cross: Tenkan crossing Kijun on 6h
        # Bullish cross: Tenkan > Kijun (and was below or equal previous bar)
        # Bearish cross: Tenkan < Kijun (and was above or equal previous bar)
        tk_bullish_cross = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
        tk_bearish_cross = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
        
        # Long logic: TK bullish cross + Kumo bullish twist
        if tk_bullish_cross and kumo_bullish:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: TK bearish cross + Kumo bearish twist
        elif tk_bearish_cross and kumo_bearish:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: long exits on TK bearish cross OR price below Kumo (Senkou B)
        elif position == 1 and (tk_bearish_cross or close[i] < senkou_b[i]):
            signals[i] = 0.0
            position = 0
        # Exit: short exits on TK bullish cross OR price above Kumo (Senkou A)
        elif position == -1 and (tk_bullish_cross or close[i] > senkou_a[i]):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_Filter_1dTrend"
timeframe = "6h"
leverage = 1.0