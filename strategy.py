#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filter_1dTrend_VolumeConfirm
Hypothesis: 6h Ichimoku TK cross with 1d cloud filter (price above/below cloud) and volume confirmation (>1.3x median). Enters long when TK crosses above in bullish cloud (price > Senkou Span A/B) with volume confirmation. Enters short when TK crosses below in bearish cloud (price < Senkou Span A/B) with volume confirmation. Uses 1d trend for cloud alignment to avoid counter-trend whipsaws. Discrete position sizing (0.25) minimizes churn. Target: 50-150 trades over 4 years. Ichimoku cloud acts as dynamic support/resistance, effective in both trending and ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Ichimoku components for 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    highest_9 = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    lowest_9 = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (highest_9 + lowest_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    highest_26 = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    lowest_26 = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (highest_26 + lowest_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    highest_52 = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    lowest_52 = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (highest_52 + lowest_52) / 2
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    # Not used for entry but confirms trend
    
    # TK Cross signals
    tk_cross_above = (tenkan > kijun) & (np.roll(tenkan, 1) <= np.roll(kijun, 1))
    tk_cross_below = (tenkan < kijun) & (np.roll(tenkan, 1) >= np.roll(kijun, 1))
    
    # Load 1d data for HTF cloud filter (aligned to 6h)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < period_senkou_b:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Ichimoku cloud components
    highest_9_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    lowest_9_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (highest_9_1d + lowest_9_1d) / 2
    
    highest_26_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    lowest_26_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (highest_26_1d + lowest_26_1d) / 2
    
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    highest_52_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    lowest_52_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (highest_52_1d + lowest_52_1d) / 2
    
    # Align 1d Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Determine cloud boundaries (Senkou Span A/B)
    upper_cloud_1d = np.maximum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    lower_cloud_1d = np.minimum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    
    # Price relative to cloud
    price_above_cloud = close > upper_cloud_1d
    price_below_cloud = close < lower_cloud_1d
    price_in_cloud = ~(price_above_cloud | price_below_cloud)
    
    # Volume confirmation: volume > 1.3x 30-period median
    volume_series = pd.Series(volume)
    vol_median = volume_series.rolling(window=30, min_periods=30).median().values
    volume_confirm = volume > (1.3 * vol_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 52-period Senkou B, 30-period volume median)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(upper_cloud_1d[i]) or np.isnan(lower_cloud_1d[i]) or 
            np.isnan(vol_median[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: TK cross above + price above cloud + volume confirmation
        if tk_cross_above[i] and price_above_cloud[i] and volume_confirm[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: TK cross below + price below cloud + volume confirmation
        elif tk_cross_below[i] and price_below_cloud[i] and volume_confirm[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: opposite TK cross or price returns to cloud
        elif position == 1 and (tk_cross_below[i] or price_below_cloud[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (tk_cross_above[i] or price_above_cloud[i]):
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

name = "6h_Ichimoku_Cloud_Filter_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0