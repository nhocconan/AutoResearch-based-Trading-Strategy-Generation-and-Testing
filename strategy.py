#!/usr/bin/env python3
# 6h_1d_ichimoku_cloud_filter_v2
# Hypothesis: 6h strategy using 1d Ichimoku cloud for trend direction and 6h price action for entries.
# Long: Price > 1d cloud (Senkou Span A/B), Tenkan > Kijun on 6h, and close > open.
# Short: Price < 1d cloud, Tenkan < Kijun on 6h, and close < open.
# Exit: Price crosses opposite cloud boundary or Tenkan/Kijun cross reverses.
# Uses 6h primary timeframe with 1d HTF for Ichimoku cloud and trend filter.
# Designed for low trade frequency (~12-30/year) to minimize fee drag while capturing major trends.
# Ichimoku cloud acts as dynamic support/resistance, working in both bull (breakouts above cloud) and bear (breakdowns below cloud) markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ichimoku_cloud_filter_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_prices = prices['open'].values
    
    # Get 1d data for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need enough data for Senkou Span B (52 periods)
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 52 periods ahead
    high_52 = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_52 + low_52) / 2.0)
    
    # Align 1d Ichimoku components to 6h (cloud is plotted ahead, so we use current values)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # 6h Tenkan/Kijun for entry signals
    high_9_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6h = (high_9_6h + low_9_6h) / 2.0
    
    high_26_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6h = (high_26_6h + low_26_6h) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup for longest indicator
        # Skip if any required data is NaN
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or
            np.isnan(close[i]) or np.isnan(open_prices[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A/B)
        upper_cloud = np.maximum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        lower_cloud = np.minimum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        # Bullish 6h Tenkan/Kijun cross
        tk_bullish_6h = tenkan_6h[i] > kijun_6h[i]
        tk_bullish_prev = i > 60 and tenkan_6h[i-1] <= kijun_6h[i-1]
        
        # Bearish 6h Tenkan/Kijun cross
        tk_bearish_6h = tenkan_6h[i] < kijun_6h[i]
        tk_bearish_prev = i > 60 and tenkan_6h[i-1] >= kijun_6h[i-1]
        
        if position == 1:  # Long position
            # Exit: Price drops below cloud OR Tenkan/Kijun cross turns bearish
            if (close[i] < lower_cloud or 
                (tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price rises above cloud OR Tenkan/Kijun cross turns bullish
            if (close[i] > upper_cloud or 
                (tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price above cloud AND bullish 6h Tenkan/Kijun cross AND bullish candle
            if (close[i] > upper_cloud and 
                tk_bullish_6h and 
                close[i] > open_prices[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price below cloud AND bearish 6h Tenkan/Kijun cross AND bearish candle
            elif (close[i] < lower_cloud and 
                  tk_bearish_6h and 
                  close[i] < open_prices[i]):
                position = -1
                signals[i] = -0.25
    
    return signals