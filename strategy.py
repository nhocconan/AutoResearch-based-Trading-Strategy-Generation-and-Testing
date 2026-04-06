#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud system with 1d filter.
# Long when Tenkan-sen > Kijun-sen AND price > Senkou Span A/B AND 1d close > 1d open.
# Short when Tenkan-sen < Kijun-sen AND price < Senkou Span A/B AND 1d close < 1d open.
# Uses Ichimoku for trend/momentum and 1d trend filter to avoid counter-trend trades.
# Target: 75-150 total trades over 4 years (19-38/year).

name = "6h_ichimoku_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Ichimoku components (9, 26, 52)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    tenkan = (high_series.rolling(window=9, min_periods=9).max() + 
              low_series.rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + low)/2
    kijun = (high_series.rolling(window=26, min_periods=26).max() + 
             low_series.rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods
    senkou_b = ((high_series.rolling(window=52, min_periods=52).max() + 
                 low_series.rolling(window=52, min_periods=52).min()) / 2).shift(26)
    
    # Current Ichimoku values (no look-ahead)
    tenkan_val = tenkan.values
    kijun_val = kijun.values
    senkou_a_val = senkou_a.values
    senkou_b_val = senkou_b.values
    
    # Daily trend filter from 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    daily_open = df_1d['open'].values
    daily_close = df_1d['close'].values
    daily_bullish = daily_close > daily_open
    daily_bearish = daily_close < daily_open
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish)
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):
        # Skip if daily data not available
        if np.isnan(daily_bullish_aligned[i]) or np.isnan(daily_bearish_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Ichimoku conditions
        price_above_cloud = (close[i] > senkou_a_val[i] and close[i] > senkou_b_val[i])
        price_below_cloud = (close[i] < senkou_a_val[i] and close[i] < senkou_b_val[i])
        tenkan_above_kijun = tenkan_val[i] > kijun_val[i]
        tenkan_below_kijun = tenkan_val[i] < kijun_val[i]
        
        # Check exits
        if position == 1:  # long position
            # Exit: price falls below cloud OR tenkan crosses below kijun
            if (not price_above_cloud or 
                tenkan_below_kijun):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price rises above cloud OR tenkan crosses above kijun
            if (not price_below_cloud or 
                tenkan_above_kijun):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with Ichimoku signals and daily trend filter
            if tenkan_above_kijun and price_above_cloud and daily_bullish_aligned[i]:
                signals[i] = 0.25
                position = 1
            elif tenkan_below_kijun and price_below_cloud and daily_bearish_aligned[i]:
                signals[i] = -0.25
                position = -1
    
    return signals