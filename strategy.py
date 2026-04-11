#!/usr/bin/env python3
# 6h_1w_icm_pullback_v1
# Strategy: 6h Ichimoku pullback with weekly cloud filter
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: In strong weekly trends (price above/below weekly Kumo), pullbacks to the 6h Tenkan-Kijun mean offer high-probability entries.
# Works in bull via long pullbacks above weekly cloud, in bear via short pullbacks below weekly cloud.
# Low trade frequency expected due to strict weekly trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_icm_pullback_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:  # Need ~1 year of weekly data
        return np.zeros(n)
    
    # Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max()
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max()
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    # We'll compute current Senkou A for cloud calculation (no shift needed for current cloud)
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max()
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = (high_52 + low_52) / 2
    
    # Weekly Ichimoku for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Tenkan (9-period)
    high_9_1w = pd.Series(high_1w).rolling(window=9, min_periods=9).max()
    low_9_1w = pd.Series(low_1w).rolling(window=9, min_periods=9).min()
    tenkan_1w = (high_9_1w + low_9_1w) / 2
    
    # Weekly Kijun (26-period)
    high_26_1w = pd.Series(high_1w).rolling(window=26, min_periods=26).max()
    low_26_1w = pd.Series(low_1w).rolling(window=26, min_periods=26).min()
    kijun_1w = (high_26_1w + low_26_1w) / 2
    
    # Weekly Senkou Span A
    senkou_a_1w = (tenkan_1w + kijun_1w) / 2
    
    # Weekly Senkou Span B (52-period)
    high_52_1w = pd.Series(high_1w).rolling(window=52, min_periods=52).max()
    low_52_1w = pd.Series(low_1w).rolling(window=52, min_periods=52).min()
    senkou_b_1w = (high_52_1w + low_52_1w) / 2
    
    # Align weekly Ichimoku components to 6h
    tenkan_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_1w.values)
    kijun_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_1w.values)
    senkou_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_a_1w.values)
    senkou_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_1w.values)
    
    # Weekly Kumo (cloud) boundaries
    weekly_kumo_top = np.maximum(senkou_a_1w_aligned, senkou_b_1w_aligned)
    weekly_kumo_bottom = np.minimum(senkou_a_1w_aligned, senkou_b_1w_aligned)
    
    # Determine if price is above or below weekly cloud
    price_above_weekly_kumo = close > weekly_kumo_top
    price_below_weekly_kumo = close < weekly_kumo_bottom
    
    # Tenkan-Kijun cross signals
    tk_cross_up = (tenkan > kijun) & (tenkan.shift(1) <= kijun.shift(1))
    tk_cross_down = (tenkan < kijun) & (tenkan.shift(1) >= kijun.shift(1))
    
    # Pullback to Tenkan-Kijun mean
    tk_mean = (tenkan + kijun) / 2
    pullback_to_tk = np.abs(close - tk_mean) < 0.5 * np.abs(tenkan - kijun)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if any required data is invalid
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(tenkan_1w_aligned[i]) or np.isnan(kijun_1w_aligned[i]) or
            np.isnan(senkou_a_1w_aligned[i]) or np.isnan(senkou_b_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry conditions
        # Long: Price above weekly cloud, TK cross up, pullback to TK mean
        if (price_above_weekly_kumo[i] and tk_cross_up[i] and pullback_to_tk[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short: Price below weekly cloud, TK cross down, pullback to TK mean
        elif (price_below_weekly_kumo[i] and tk_cross_down[i] and pullback_to_tk[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: Opposite TK cross
        elif position == 1 and tk_cross_down[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and tk_cross_up[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals