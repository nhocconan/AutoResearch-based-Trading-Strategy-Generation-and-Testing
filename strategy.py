#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_WeeklyTrend
Hypothesis: Trade 6h timeframe using Ichimoku cloud breakouts filtered by weekly trend direction.
Enter long when price breaks above Ichimoku cloud (Senkou Span A/B) AND weekly trend is bullish (price > weekly EMA50).
Enter short when price breaks below Ichimoku cloud AND weekly trend is bearish (price < weekly EMA50).
Exit when price re-enters the cloud or weekly trend reverses.
Uses discrete sizing 0.25 to manage risk. Target 15-35 trades/year on 6h timeframe.
Ichimoku cloud provides dynamic support/resistance that adapts to volatility.
Weekly EMA50 filter ensures we only trade with the higher timeframe trend, improving performance in both bull and bear markets by avoiding counter-trend whipsaws.
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, plotted 26 periods ahead
    senkou_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, plotted 26 periods ahead
    period_senkou_b = 52
    high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (high_senkou_b + low_senkou_b) / 2
    
    # The actual cloud boundaries at current time are Senkou Span A/B from 26 periods ago
    # We need to shift Senkou A/B back by 26 periods to get current cloud
    senkou_a_lagged = np.roll(senkou_a, 26)
    senkou_b_lagged = np.roll(senkou_b, 26)
    # Fill the first 26 values with NaN since they don't have valid cloud data
    senkou_a_lagged[:26] = np.nan
    senkou_b_lagged[:26] = np.nan
    
    # Upper cloud boundary is max of Senkou A/B, lower cloud boundary is min
    upper_cloud = np.maximum(senkou_a_lagged, senkou_b_lagged)
    lower_cloud = np.minimum(senkou_a_lagged, senkou_b_lagged)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for weekly EMA50 (50) and Ichimoku (52 + 26 shift)
    start_idx = max(50, 52 + 26)  # 78
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper cloud AND weekly trend bullish (price > weekly EMA50)
            long_setup = (close[i] > upper_cloud[i]) and \
                         (close[i] > ema_50_1w_aligned[i])
            # Short: price breaks below lower cloud AND weekly trend bearish (price < weekly EMA50)
            short_setup = (close[i] < lower_cloud[i]) and \
                          (close[i] < ema_50_1w_aligned[i])
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price re-enters cloud (closes below upper cloud) OR weekly trend turns bearish
            if (close[i] < upper_cloud[i]) or \
               (close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters cloud (closes above lower cloud) OR weekly trend turns bullish
            if (close[i] > lower_cloud[i]) or \
               (close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_WeeklyTrend"
timeframe = "6h"
leverage = 1.0