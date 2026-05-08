#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter
# Ichimoku (Tenkan/Kijun/Senkou) identifies trend, support/resistance, and momentum.
# Price above/below cloud indicates trend direction. TK cross provides entry signals.
# Using 1d ADX > 25 to filter for strong trends only, avoiding whipsaws in ranges.
# This works in both bull/bear markets by trading with the dominant trend on higher timeframe.
# Targets 15-25 trades per year (~60-100 total over 4 years) to minimize fee drag.

name = "6h_Ichimoku_Cloud_1dADX"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Ichimoku and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).max().values
    
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).min().values
    
    tenkan_sen = (rolling_max(high_1d, 9) + rolling_min(low_1d, 9)) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun_sen = (rolling_max(high_1d, 26) + rolling_min(low_1d, 26)) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    senkou_span_b = (rolling_max(high_1d, 52) + rolling_min(low_1d, 52)) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # ADX trend filter on 1d
    # Calculate ADX(14) on daily
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(high_1d)
    tr = np.zeros_like(high_1d)
    
    for i in range(1, len(high_1d)):
        plus_dm[i] = max(high_1d[i] - high_1d[i-1], 0)
        minus_dm[i] = max(low_1d[i-1] - low_1d[i], 0)
        if plus_dm[i] == minus_dm[i]:
            plus_dm[i] = 0
            minus_dm[i] = 0
        tr[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    # Wilder smoothing
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    tr14 = wilder_smooth(tr, 14)
    plus_dm14 = wilder_smooth(plus_dm, 14)
    minus_dm14 = wilder_smooth(minus_dm, 14)
    
    # Avoid division by zero
    plus_di14 = np.where(tr14 != 0, 100 * (plus_dm14 / tr14), 0)
    minus_di14 = np.where(tr14 != 0, 100 * (minus_dm14 / tr14), 0)
    
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 0)
    adx = wilder_smooth(dx, 14)
    
    adx_strong = adx > 25
    adx_strong_6h = align_htf_to_ltf(prices, df_1d, adx_strong)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Ensure sufficient data for Ichimoku
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or 
            np.isnan(adx_strong_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_6h[i], senkou_b_6h[i])
        cloud_bottom = min(senkou_a_6h[i], senkou_b_6h[i])
        
        if position == 0:
            # Enter long: price above cloud, TK cross bullish, strong trend
            if (close[i] > cloud_top and 
                tenkan_6h[i] > kijun_6h[i] and 
                adx_strong_6h[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price below cloud, TK cross bearish, strong trend
            elif (close[i] < cloud_bottom and 
                  tenkan_6h[i] < kijun_6h[i] and 
                  adx_strong_6h[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to cloud or TK cross bearish
            if close[i] < cloud_bottom or tenkan_6h[i] < kijun_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to cloud or TK cross bullish
            if close[i] > cloud_top or tenkan_6h[i] > kijun_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals