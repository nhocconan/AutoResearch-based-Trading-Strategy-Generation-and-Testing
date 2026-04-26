#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filter_1dTrend_v1
Hypothesis: 6h Ichimoku system with 1d trend filter for BTC/ETH.
- Uses 6h timeframe targeting 50-150 total trades over 4 years (12-37/year)
- Long when: Tenkan > Kijun, price > Cloud (Senkou Span A/B), and 1d uptrend (price > 1d EMA50)
- Short when: Tenkan < Kijun, price < Cloud, and 1d downtrend (price < 1d EMA50)
- Ichimoku provides multiple confluence factors (trend, momentum, support/resistance)
- 1d EMA50 filter ensures alignment with higher timeframe trend to avoid counter-trend whipsaws
- Designed for lower frequency with proven Ichimoku edge in crypto markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for Ichimoku calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
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
    
    # Align Ichimoku components to 6h timeframe (they're already calculated on 6h)
    tenkan_aligned = tenkan  # Already on 6h
    kijun_aligned = kijun    # Already on 6h
    senkou_a_aligned = senkou_a  # Already on 6h
    senkou_b_aligned = senkou_b  # Already on 6h
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 52 for Senkou B)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(ema50_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Ichimoku entry conditions with 1d trend filter
        if position == 0:
            # Long: Tenkan > Kijun AND price > Cloud AND 1d uptrend
            if (tenkan_aligned[i] > kijun_aligned[i] and 
                close[i] > cloud_top and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Tenkan < Kijun AND price < Cloud AND 1d downtrend
            elif (tenkan_aligned[i] < kijun_aligned[i] and 
                  close[i] < cloud_bottom and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Tenkan < Kijun OR price < Cloud
            if tenkan_aligned[i] < kijun_aligned[i] or close[i] < cloud_bottom:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Tenkan > Kijun OR price > Cloud
            if tenkan_aligned[i] > kijun_aligned[i] or close[i] > cloud_top:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Filter_1dTrend_v1"
timeframe = "6h"
leverage = 1.0