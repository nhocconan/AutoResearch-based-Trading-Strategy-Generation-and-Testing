#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with Weekly Trend Filter and Volume Spike Confirmation.
- Primary timeframe: 6h for lower trade frequency (target: 50-150 trades over 4 years).
- HTF: 1w for trend direction (bullish if price > weekly cloud, bearish if price < weekly cloud).
- Ichimoku Components (calculated on 6h): Tenkan-sen (9), Kijun-sen (26), Senkou Span A/B (26, 52 displaced).
- Entry: Long when Tenkan crosses above Kijun AND price > weekly cloud AND volume spike.
         Short when Tenkan crosses below Kijun AND price < weekly cloud AND volume spike.
- Exit: Reverse Tenkan/Kijun cross OR loss of volume confirmation.
- Signal size: 0.25 discrete to minimize fee churn and control drawdown.
- Weekly cloud provides major support/resistance; Ichimoku TK cross captures momentum with trend filter.
- Volume spike confirms institutional participation. Works in bull/bear by only trading with weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly trend filter (cloud)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_9 = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_26 = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_span_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    max_high_52 = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = (max_high_52 + min_low_52) / 2
    
    # Calculate weekly cloud (Senkou Span A/B from 1w)
    # Weekly Tenkan-sen (9)
    wk_max_high_9 = pd.Series(df_1w['high']).rolling(window=9, min_periods=9).max().values
    wk_min_low_9 = pd.Series(df_1w['low']).rolling(window=9, min_periods=9).min().values
    wk_tenkan = (wk_max_high_9 + wk_min_low_9) / 2
    
    # Weekly Kijun-sen (26)
    wk_max_high_26 = pd.Series(df_1w['high']).rolling(window=26, min_periods=26).max().values
    wk_min_low_26 = pd.Series(df_1w['low']).rolling(window=26, min_periods=26).min().values
    wk_kijun = (wk_max_high_26 + wk_min_low_26) / 2
    
    # Weekly Senkou Span A
    wk_senkou_span_a = (wk_tenkan + wk_kijun) / 2
    
    # Weekly Senkou Span B (52)
    wk_max_high_52 = pd.Series(df_1w['high']).rolling(window=52, min_periods=52).max().values
    wk_min_low_52 = pd.Series(df_1w['low']).rolling(window=52, min_periods=52).min().values
    wk_senkou_span_b = (wk_max_high_52 + wk_min_low_52) / 2
    
    # Weekly Cloud: between Senkou Span A and B
    weekly_cloud_top = np.maximum(wk_senkou_span_a, wk_senkou_span_b)
    weekly_cloud_bottom = np.minimum(wk_senkou_span_a, wk_senkou_span_b)
    
    # Align HTF indicators to 6h
    weekly_cloud_top_aligned = align_htf_to_ltf(prices, df_1w, weekly_cloud_top)
    weekly_cloud_bottom_aligned = align_htf_to_ltf(prices, df_1w, weekly_cloud_bottom)
    
    # Volume confirmation: current 6h volume > 2.0 * 20-period volume MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(52, 20, 26)  # Need enough bars for Senkou Span B (52), volume MA (20), Kijun (26)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(weekly_cloud_top_aligned[i]) or np.isnan(weekly_cloud_bottom_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Check for TK cross signals with volume spike and weekly trend filter
            if volume_spike[i]:
                # Bullish TK cross: Tenkan crosses above Kijun
                bullish_cross = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
                # Price above weekly cloud (bullish trend)
                above_cloud = curr_close > weekly_cloud_top_aligned[i]
                
                if bullish_cross and above_cloud:
                    signals[i] = 0.25
                    position = 1
                # Bearish TK cross: Tenkan crosses below Kijun
                bearish_cross = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
                # Price below weekly cloud (bearish trend)
                below_cloud = curr_close < weekly_cloud_bottom_aligned[i]
                
                if bearish_cross and below_cloud:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Tenkan crosses below Kijun OR loss of volume confirmation OR price falls below weekly cloud
            if (tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]) or \
               not volume_spike[i] or \
               curr_close < weekly_cloud_bottom_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Tenkan crosses above Kijun OR loss of volume confirmation OR price rises above weekly cloud
            if (tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]) or \
               not volume_spike[i] or \
               curr_close > weekly_cloud_top_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_WeeklyCloud_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0