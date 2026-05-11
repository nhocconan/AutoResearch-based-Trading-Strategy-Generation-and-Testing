#!/usr/bin/env python3
name = "6h_Ichimoku_Cloud_Kumo_Twist_12hTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 12h data for Ichimoku components
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 52:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_12h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_12h).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_12h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_12h).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_12h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_12h).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h
    tenkan_6h = align_htf_to_ltf(prices, df_12h, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_12h, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_12h, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_12h, senkou_b)
    
    # Kumo twist detection: Senkou A crosses Senkou B
    # Bullish twist: Senkou A crosses above Senkou B
    # Bearish twist: Senkou A crosses below Senkou B
    senkou_a_prev = np.roll(senkou_a_6h, 1)
    senkou_b_prev = np.roll(senkou_b_6h, 1)
    senkou_a_prev[0] = np.nan
    senkou_b_prev[0] = np.nan
    
    bullish_twist = (senkou_a_6h > senkou_b_6h) & (senkou_a_prev <= senkou_b_prev)
    bearish_twist = (senkou_a_6h < senkou_b_6h) & (senkou_a_prev >= senkou_b_prev)
    
    # TK Cross: Tenkan crosses Kijun
    tenkan_prev = np.roll(tenkan_6h, 1)
    kijun_prev = np.roll(kijun_6h, 1)
    tenkan_prev[0] = np.nan
    kijun_prev[0] = np.nan
    
    tk_bullish = (tenkan_6h > kijun_6h) & (tenkan_prev <= kijun_prev)
    tk_bearish = (tenkan_6h < kijun_6h) & (tenkan_prev >= kijun_prev)
    
    # Price vs Cloud: price above/below both Senkou spans
    price_above_cloud = (close > senkou_a_6h) & (close > senkou_b_6h)
    price_below_cloud = (close < senkou_a_6h) & (close < senkou_b_6h)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 52  # Ensure Ichimoku is ready
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bullish TK cross + price above cloud + Kumo bullish twist
            if (tk_bullish[i] and price_above_cloud[i] and bullish_twist[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish TK cross + price below cloud + Kumo bearish twist
            elif (tk_bearish[i] and price_below_cloud[i] and bearish_twist[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bearish TK cross or price falls below cloud
            if tk_bearish[i] or not price_above_cloud[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bullish TK cross or price rises above cloud
            if tk_bullish[i] or not price_below_cloud[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals