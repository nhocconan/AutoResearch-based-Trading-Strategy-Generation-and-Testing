#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with Weekly Trend Filter
# Uses daily Tenkan/Kijun cross for entry, with weekly Cloud (Senkou Span A/B) as trend filter.
# Long when price > Cloud and Tenkan > Kijun; Short when price < Cloud and Tenkan < Kijun.
# Works in bull markets (trend-following above cloud) and bear markets (trend-following below cloud).
# Target: 50-150 total trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load daily data for Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Load weekly data for trend filter (Cloud)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Ichimoku calculations (daily)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Weekly Cloud (for trend filter)
    # Weekly Senkou Span A: (weekly Tenkan + weekly Kijun) / 2
    ws9_high = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    ws9_low = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    w_tenkan = (ws9_high + ws9_low) / 2
    
    ws26_high = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    ws26_low = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    w_kijun = (ws26_high + ws26_low) / 2
    
    w_senkou_a = (w_tenkan + w_kijun) / 2
    
    # Weekly Senkou Span B: (52-week high + 52-week low) / 2
    ws52_high = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    ws52_low = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    w_senkou_b = (ws52_high + ws52_low) / 2
    
    # Align all indicators to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    w_senkou_a_6h = align_htf_to_ltf(prices, df_1w, w_senkou_a)
    w_senkou_b_6h = align_htf_to_ltf(prices, df_1w, w_senkou_b)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or
            np.isnan(w_senkou_a_6h[i]) or np.isnan(w_senkou_b_6h[i])):
            continue
        
        # Cloud boundaries (weekly)
        cloud_top = np.maximum(w_senkou_a_6h[i], w_senkou_b_6h[i])
        cloud_bottom = np.minimum(w_senkou_a_6h[i], w_senkou_b_6h[i])
        
        # Long entry: price above cloud + Tenkan > Kijun
        if (close[i] > cloud_top and tenkan_6h[i] > kijun_6h[i] and position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price below cloud + Tenkan < Kijun
        elif (close[i] < cloud_bottom and tenkan_6h[i] < kijun_6h[i] and position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite Tenkan/Kijun cross
        elif position == 1 and tenkan_6h[i] < kijun_6h[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and tenkan_6h[i] > kijun_6h[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_Ichimoku_Cloud_WeeklyTrend"
timeframe = "6h"
leverage = 1.0