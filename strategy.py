#!/usr/bin/env python3
# 6h_Ichimoku_Cloud_Breakout_1wTrend
# Hypothesis: Ichimoku Cloud (Tenkan/Kijun) breakout with weekly trend filter (price > weekly Kumo) captures multi-timeframe momentum.
# Works in bull/bear: weekly trend filter ensures alignment with higher timeframe direction, reducing whipsaws.
# Entry: Price breaks above/below Kumo cloud with Tenkan/Kijun cross in same direction, confirmed by volume spike.
# Exit: Price re-enters Kumo cloud or Tenkan/Kijun reverse cross.
# Target: 50-150 total trades over 4 years (~12-37/year) via strict Ichimoku + weekly trend + volume confluence.

timeframe = "6h"
name = "6h_Ichimoku_Cloud_Breakout_1wTrend"
leverage = 1.0

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
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    # Kumo (Cloud): between Senkou Span A and B
    
    # Calculate Tenkan-sen (9-period)
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Calculate Kijun-sen (26-period)
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Calculate Senkou Span A
    senkou_a = (tenkan + kijun) / 2
    
    # Calculate Senkou Span B (52-period)
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # Weekly trend filter: price > weekly Kumo (cloud)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly Ichimoku components for trend filter
    wh_high = df_1w['high'].values
    wh_low = df_1w['low'].values
    wh_close = df_1w['close'].values
    
    # Weekly Tenkan (9-period)
    wh_high_9 = pd.Series(wh_high).rolling(window=9, min_periods=9).max().values
    wh_low_9 = pd.Series(wh_low).rolling(window=9, min_periods=9).min().values
    w_tenkan = (wh_high_9 + wh_low_9) / 2
    
    # Weekly Kijun (26-period)
    wh_high_26 = pd.Series(wh_high).rolling(window=26, min_periods=26).max().values
    wh_low_26 = pd.Series(wh_low).rolling(window=26, min_periods=26).min().values
    w_kijun = (wh_high_26 + wh_low_26) / 2
    
    # Weekly Senkou Span A
    w_senkou_a = (w_tenkan + w_kijun) / 2
    
    # Weekly Senkou Span B (52-period)
    wh_high_52 = pd.Series(wh_high).rolling(window=52, min_periods=52).max().values
    wh_low_52 = pd.Series(wh_low).rolling(window=52, min_periods=52).min().values
    w_senkou_b = (wh_high_52 + wh_low_52) / 2
    
    # Weekly trend: price above both Senkou Spans (bullish) or below both (bearish)
    w_kumo_top = np.maximum(w_senkou_a, w_senkou_b)
    w_kumo_bottom = np.minimum(w_senkou_a, w_senkou_b)
    
    # Align weekly Ichimoku components to 6h
    w_tenkan_aligned = align_htf_to_ltf(prices, df_1w, w_tenkan)
    w_kijun_aligned = align_htf_to_ltf(prices, df_1w, w_kijun)
    w_kumo_top_aligned = align_htf_to_ltf(prices, df_1w, w_kumo_top)
    w_kumo_bottom_aligned = align_htf_to_ltf(prices, df_1w, w_kumo_bottom)
    
    # Volume spike: 2x average volume (48-period = 2 days on 6h chart)
    vol_ma = pd.Series(volume).rolling(window=48, min_periods=48).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 48)  # Ensure we have Ichimoku and volume data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or 
            np.isnan(senkou_b[i]) or np.isnan(w_tenkan_aligned[i]) or np.isnan(w_kijun_aligned[i]) or
            np.isnan(w_kumo_top_aligned[i]) or np.isnan(w_kumo_bottom_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current cloud boundaries
        kumo_top = max(senkou_a[i], senkou_b[i])
        kumo_bottom = min(senkou_a[i], senkou_b[i])
        
        # Ichimoku signals
        tenkan_cross_above_kijun = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
        tenkan_cross_below_kijun = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
        price_above_kumo = close[i] > kumo_top
        price_below_kumo = close[i] < kumo_bottom
        
        if position == 0:
            # Long: price breaks above cloud with Tenkan/Kijun cross up, weekly bullish trend, volume spike
            if (price_above_kumo and tenkan_cross_above_kijun and 
                close[i] > w_kumo_top_aligned[i] and volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below cloud with Tenkan/Kijun cross down, weekly bearish trend, volume spike
            elif (price_below_kumo and tenkan_cross_below_kijun and 
                  close[i] < w_kumo_bottom_aligned[i] and volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price re-enters cloud or Tenkan/Kijun cross down
            if close[i] < kumo_top or tenkan_cross_below_kijun:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price re-enters cloud or Tenkan/Kijun cross up
            if close[i] > kumo_bottom or tenkan_cross_above_kijun:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals