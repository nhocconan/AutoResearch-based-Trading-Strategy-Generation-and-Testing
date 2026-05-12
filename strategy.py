#!/usr/bin/env python3
# 6h_Ichimoku_Cloud_TK_Cross_1dTrendFilter
# Hypothesis: On 6h timeframe, use Ichimoku cloud from daily timeframe for trend direction, with Tenkan/Kijun cross on 6h for entry timing.
# Long when price > daily Ichimoku cloud AND Tenkan crosses above Kijun on 6h.
# Short when price < daily Ichimoku cloud AND Tenkan crosses below Kijun on 6h.
# Exit when price crosses back into the cloud or Tenkan/Kijun cross reverses.
# Uses daily timeframe for trend filter (cloud) to avoid false signals in choppy markets.
# Ichimoku is effective in both trending and ranging markets, providing clear support/resistance.
# Targets 15-30 trades/year for low fee drag and works in both bull and bear markets by following the higher timeframe trend.

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
    
    # Calculate Ichimoku components on daily timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = pd.Series(daily_high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(daily_low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period26_high = pd.Series(daily_high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(daily_low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period52_high = pd.Series(daily_high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(daily_low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Chikou Span (Lagging Span): current close plotted 26 periods back
    # Not used for signals but calculated for completeness
    
    # Cloud is between Senkou A and Senkou B
    # Upper cloud = max(senkou_a, senkou_b)
    # Lower cloud = min(senkou_a, senkou_b)
    upper_cloud = np.maximum(senkou_a, senkou_b)
    lower_cloud = np.minimum(senkou_a, senkou_b)
    
    # Calculate Tenkan/Kijun cross on 6h timeframe
    period9_high_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6h = (period9_high_6h + period9_low_6h) / 2
    
    period26_high_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6h = (period26_high_6h + period26_low_6h) / 2
    
    # Align all daily Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    upper_cloud_aligned = align_htf_to_ltf(prices, df_1d, upper_cloud)
    lower_cloud_aligned = align_htf_to_ltf(prices, df_1d, lower_cloud)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(upper_cloud_aligned[i]) or np.isnan(lower_cloud_aligned[i]) or
            np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Ichimoku conditions
        price_above_cloud = close[i] > upper_cloud_aligned[i]
        price_below_cloud = close[i] < lower_cloud_aligned[i]
        tenkan_above_kijun_6h = tenkan_6h[i] > kijun_6h[i]
        tenkan_below_kijun_6h = tenkan_6h[i] < kijun_6h[i]
        
        if position == 0:
            # LONG: Price above cloud AND Tenkan crosses above Kijun (bullish cross)
            if price_above_cloud and tenkan_above_kijun_6h and tenkan_6h[i-1] <= kijun_6h[i-1]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below cloud AND Tenkan crosses below Kijun (bearish cross)
            elif price_below_cloud and tenkan_below_kijun_6h and tenkan_6h[i-1] >= kijun_6h[i-1]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price enters cloud OR Tenkan crosses below Kijun
            if not price_above_cloud or tenkan_below_kijun_6h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price enters cloud OR Tenkan crosses above Kijun
            if not price_below_cloud or tenkan_above_kijun_6h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals