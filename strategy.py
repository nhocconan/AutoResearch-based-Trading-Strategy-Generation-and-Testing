#!/usr/bin/env python3
# 6h_Ichimoku_Cloud_TK_Cross_1dTrendFilter
# Hypothesis: On 6h timeframe, enter long when Tenkan-sen crosses above Kijun-sen and price is above the Kumo (cloud), with 1d trend filter (price > 1d EMA50).
# Enter short when Tenkan-sen crosses below Kijun-sen and price is below the Kumo, with 1d trend filter (price < 1d EMA50).
# Exit when Tenkan-sen and Kijun-sen cross in the opposite direction.
# Uses Ichimoku cloud for trend and momentum, with daily trend filter to avoid counter-trend trades.
# Designed to work in both bull and bear markets by aligning with higher timeframe trend.

name = "6h_Ichimoku_Cloud_TK_Cross_1dTrendFilter"
timeframe = "6h"
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
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2.0
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_ema50 = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_ema50_aligned = align_htf_to_ltf(prices, df_1d, daily_ema50)
    
    # Align Ichimoku components to 6h timeframe (no additional delay needed as they are based on current close)
    tenkan_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), tenkan)
    kijun_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), senkou_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure Ichimoku is stable (max period 52 + buffer)
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(daily_ema50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        daily_trend = daily_ema50_aligned[i]
        
        # Kumo (cloud) boundaries: Senkou Span A and B
        upper_cloud = max(senkou_a_val, senkou_b_val)
        lower_cloud = min(senkou_a_val, senkou_b_val)
        
        if position == 0:
            # LONG: TK cross up + price above cloud + daily uptrend
            if (tenkan_val > kijun_val and 
                tenkan_aligned[i-1] <= kijun_aligned[i-1] and  # crossed up on this bar
                close[i] > upper_cloud and 
                close[i] > daily_trend):
                signals[i] = 0.25
                position = 1
            # SHORT: TK cross down + price below cloud + daily downtrend
            elif (tenkan_val < kijun_val and 
                  tenkan_aligned[i-1] >= kijun_aligned[i-1] and  # crossed down on this bar
                  close[i] < lower_cloud and 
                  close[i] < daily_trend):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TK cross down (regardless of cloud)
            if tenkan_val < kijun_val and tenkan_aligned[i-1] >= kijun_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TK cross up (regardless of cloud)
            if tenkan_val > kijun_val and tenkan_aligned[i-1] <= kijun_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals