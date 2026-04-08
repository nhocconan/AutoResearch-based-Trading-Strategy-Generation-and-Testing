#!/usr/bin/env python3
# 6h_1d_ichimoku_kumo_twist_v1
# Hypothesis: 6-hour Ichimoku with daily Kumo twist filter. Long when price above Kumo (1d) and TK cross bullish (6h), short when price below Kumo and TK cross bearish.
# Uses daily Kumo as macro trend filter and 6h TK cross for entry timing to avoid whipsaw.
# Target: 20-40 trades/year to minimize fee drag while capturing sustained trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ichimoku_kumo_twist_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate 6h Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    max_high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (max_high_senkou_b + min_low_senkou_b) / 2
    
    # Get 1d data for Kumo (cloud)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily Tenkan-sen (9-period)
    max_high_tenkan_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    min_low_tenkan_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (max_high_tenkan_1d + min_low_tenkan_1d) / 2
    
    # Daily Kijun-sen (26-period)
    max_high_kijun_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    min_low_kijun_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (max_high_kijun_1d + min_low_kijun_1d) / 2
    
    # Daily Senkou Span A
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    # Daily Senkou Span B (52-period)
    max_high_senkou_b_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    min_low_senkou_b_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (max_high_senkou_b_1d + min_low_senkou_b_1d) / 2
    
    # Align daily Kumo to 6h timeframe
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Need enough data for Senkou B
        # Skip if any values are NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Kumo (cloud) boundaries from daily
        upper_kumo = max(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        lower_kumo = min(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        # TK Cross signals
        tk_cross_bullish = tenkan[i] > kijun[i]
        tk_cross_bearish = tenkan[i] < kijun[i]
        
        if position == 1:  # Long
            # Exit: price below Kumo OR TK cross bearish
            if close[i] < lower_kumo or tk_cross_bearish:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price above Kumo OR TK cross bullish
            if close[i] > upper_kumo or tk_cross_bullish:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry: price above Kumo AND TK cross bullish (long)
            # Entry: price below Kumo AND TK cross bearish (short)
            if close[i] > upper_kumo and tk_cross_bullish:
                position = 1
                signals[i] = 0.25
            elif close[i] < lower_kumo and tk_cross_bearish:
                position = -1
                signals[i] = -0.25
    
    return signals