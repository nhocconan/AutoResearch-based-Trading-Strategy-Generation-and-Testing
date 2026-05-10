#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1wTrend
Hypothesis: Combines Ichimoku cloud from daily timeframe with weekly trend filter and volume confirmation on 6h chart.
- Uses daily Ichimoku cloud (Tenkan/Kijun/Senkou) to identify trend and support/resistance
- Weekly trend filter (price above/below weekly Kumo) ensures alignment with higher timeframe trend
- Volume confirmation (current volume > 1.5x 6-day average) filters false breakouts
- Works in bull markets via buying breakouts above cloud in uptrends
- Works in bear markets via selling breakdowns below cloud in downtrends
- Designed for low trade frequency (target: 50-150 trades over 4 years) to minimize fee drag
"""

name = "6h_Ichimoku_Cloud_Breakout_1wTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components (standard periods: 9, 26, 52)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # Align Ichimoku components to 6h timeframe (shifted by 1 for proper alignment)
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Ichimoku components for trend filter
    high_9_w = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    low_9_w = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_w = (high_9_w + low_9_w) / 2
    
    high_26_w = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    low_26_w = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_w = (high_26_w + low_26_w) / 2
    
    senkou_a_w = (tenkan_w + kijun_w) / 2
    
    high_52_w = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    low_52_w = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_b_w = (high_52_w + low_52_w) / 2
    
    # Align weekly Ichimoku components to 6h timeframe
    senkou_a_w_6h = align_htf_to_ltf(prices, df_1w, senkou_a_w)
    senkou_b_w_6h = align_htf_to_ltf(prices, df_1w, senkou_b_w)
    
    # 6h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate volume filter: current volume > 1.5x 6-period average
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    volume_filter = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Ichimoku (52) and volume filter (6)
    start_idx = max(52, 6)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or
            np.isnan(senkou_a_w_6h[i]) or np.isnan(senkou_b_w_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud boundaries (Senkou Span A and B)
        upper_cloud = np.maximum(senkou_a_6h[i], senkou_b_6h[i])
        lower_cloud = np.minimum(senkou_a_6h[i], senkou_b_6h[i])
        
        # Weekly trend filter: price relative to weekly cloud
        weekly_upper_cloud = np.maximum(senkou_a_w_6h[i], senkou_b_w_6h[i])
        weekly_lower_cloud = np.minimum(senkou_a_w_6h[i], senkou_b_w_6h[i])
        price_above_weekly_cloud = close[i] > weekly_upper_cloud
        price_below_weekly_cloud = close[i] < weekly_lower_cloud
        
        # Ichimoku signals
        tk_cross_bullish = tenkan_6h[i] > kijun_6h[i]
        tk_cross_bearish = tenkan_6h[i] < kijun_6h[i]
        price_above_cloud = close[i] > upper_cloud
        price_below_cloud = close[i] < lower_cloud
        
        if position == 0:
            # Long entry: price breaks above cloud + TK cross bullish + weekly uptrend + volume
            if price_above_cloud and tk_cross_bullish and price_above_weekly_cloud and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below cloud + TK cross bearish + weekly downtrend + volume
            elif price_below_cloud and tk_cross_bearish and price_below_weekly_cloud and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below cloud or TK cross bearish
            if price_below_cloud or not tk_cross_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above cloud or TK cross bullish
            if price_above_cloud or not tk_cross_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals