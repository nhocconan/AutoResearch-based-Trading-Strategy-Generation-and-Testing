#!/usr/bin/env python3
# 6h_weekly_ichimoku_trend_v1
# Hypothesis: 6h strategy using weekly Ichimoku cloud (from 1w HTF) for trend direction and support/resistance,
# combined with 6h Tenkan-Kijun cross for entry timing and volume confirmation (>1.5x 20-bar avg volume).
# Enters long when price is above weekly cloud, Tenkan crosses above Kijun, and volume confirms;
# enters short when price is below weekly cloud, Tenkan crosses below Kijun, and volume confirms.
# Exits on opposite Tenkan/Kijun cross or when price re-enters the weekly cloud.
# Uses discrete sizing (0.25) to limit fee churn. Target: 12-37 trades/year (50-150 total over 4 years).
# Weekly Ichimoku provides multi-timeframe trend structure that works in bull/bear markets;
# Tenkan-Kijun cross gives timely entries; volume filters breakout strength.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_ichimoku_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period = ~5 days of 6h bars)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 6h Tenkan-sen (9-period) and Kijun-sen (26-period)
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Multi-timeframe: weekly Ichimoku cloud (from 1w HTF)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Tenkan-sen (9-period) and Kijun-sen (26-period)
    high_9_1w = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    low_9_1w = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_1w = (high_9_1w + low_9_1w) / 2
    
    high_26_1w = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    low_26_1w = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_1w = (high_26_1w + low_26_1w) / 2
    
    # Weekly Senkou Span A and B (leading span)
    senkou_a_1w = ((tenkan_1w + kijun_1w) / 2)
    high_52_1w = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    low_52_1w = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_b_1w = (high_52_1w + low_52_1w) / 2
    
    # Align weekly Ichimoku components to 6h timeframe (wait for weekly close)
    tenkan_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_1w)
    kijun_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_1w)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_a_1w)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_1w)
    
    # Weekly cloud boundaries
    weekly_cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    weekly_cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(tenkan_1w_aligned[i]) or np.isnan(kijun_1w_aligned[i]) or
            np.isnan(weekly_cloud_top[i]) or np.isnan(weekly_cloud_bottom[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Tenkan/Kijun cross signals
        tenkan_prev = tenkan[i-1] if i > 0 else tenkan[i]
        kijun_prev = kijun[i-1] if i > 0 else kijun[i]
        tk_cross_up = (tenkan[i] > kijun[i]) and (tenkan_prev <= kijun_prev)
        tk_cross_down = (tenkan[i] < kijun[i]) and (tenkan_prev >= kijun_prev)
        
        if position == 1:  # Long position
            # Exit: Tenkan crosses below Kijun OR price re-enters weekly cloud (below cloud top)
            if tk_cross_down or close[i] < weekly_cloud_top[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Tenkan crosses above Kijun OR price re-enters weekly cloud (above cloud bottom)
            if tk_cross_up or close[i] > weekly_cloud_bottom[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for entry: price outside weekly cloud + TK cross + volume confirmation
            price_above_cloud = close[i] > weekly_cloud_top[i]
            price_below_cloud = close[i] < weekly_cloud_bottom[i]
            
            bullish_setup = price_above_cloud and tk_cross_up and volume_confirmed
            bearish_setup = price_below_cloud and tk_cross_down and volume_confirmed
            
            if bullish_setup:
                position = 1
                signals[i] = 0.25
            elif bearish_setup:
                position = -1
                signals[i] = -0.25
    
    return signals