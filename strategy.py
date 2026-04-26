#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrendFilter_v1
Hypothesis: 6h Ichimoku Kumo twist (Tenkan/Kijun cross) with 1d trend filter (price > 1d EMA50 for long, < for short).
Enters long when Tenkan crosses above Kijun AND price > Kumo (bullish twist) AND close > 1d EMA50.
Enters short when Tenkan crosses below Kijun AND price < Kumo (bearish twist) AND close < 1d EMA50.
Exits on opposite Kumo twist or when price re-enters Kumo (cloud).
Ichimoku provides dynamic support/resistance and trend identification; Kumo twist captures momentum shifts.
1d EMA50 filter ensures alignment with higher timeframe trend to avoid counter-trend trades.
Targets 12-37 trades/year (50-150 total over 4 years) on 6h timeframe.
Works in bull/bear markets by trading with the 1d trend and using Ichimoku's adaptive cloud.
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
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
    
    # Kumo (Cloud) boundaries: Senkou Span A and B
    # Kumo top is max(senkou_a, senkou_b), bottom is min(senkou_a, senkou_b)
    kumo_top = np.maximum(senkou_a, senkou_b)
    kumo_bottom = np.minimum(senkou_a, senkou_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 52 for Senkou B)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Kumo twist detection
        # Bullish twist: Tenkan crosses above Kijun AND price > Kumo
        # Bearish twist: Tenkan crosses below Kijun AND price < Kumo
        bullish_twist = (tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1] and close[i] > kumo_top[i])
        bearish_twist = (tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1] and close[i] < kumo_bottom[i])
        
        # Price re-entering Kumo (exit condition)
        price_in_kumo = (close[i] > kumo_bottom[i] and close[i] < kumo_top[i])
        
        if position == 0:
            # Long: bullish Kumo twist AND price > Kumo AND close > 1d EMA50
            if bullish_twist and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish Kumo twist AND price < Kumo AND close < 1d EMA50
            elif bearish_twist and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: bearish Kumo twist OR price re-enters Kumo
            if bearish_twist or price_in_kumo:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: bullish Kumo twist OR price re-enters Kumo
            if bullish_twist or price_in_kumo:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrendFilter_v1"
timeframe = "6h"
leverage = 1.0