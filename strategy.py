#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1w_1d_ichimoku_trend_v1
# Uses Ichimoku Cloud on daily timeframe with weekly trend filter.
# Long when price above cloud and Tenkan > Kijun with weekly uptrend (price above weekly Senkou Span B).
# Short when price below cloud and Tenkan < Kijun with weekly downtrend (price below weekly Senkou Span B).
# Ichimoku provides dynamic support/resistance and trend direction, effective in both bull and bear markets.
# Weekly filter ensures we only trade in the direction of higher timeframe trend, reducing whipsaw.
# Target: 15-30 trades/year per symbol for low friction and high edge.

name = "6h_1w_1d_ichimoku_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 26:
        return np.zeros(n)
    
    # Calculate Ichimoku components on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_9 = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_26 = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    max_high_52 = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (max_high_52 + min_low_52) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Weekly trend filter: price relative to weekly Senkou Span B
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    max_high_52w = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    min_low_52w = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_b_1w = (max_high_52w + min_low_52w) / 2
    
    senkou_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_1w)
    
    # Determine cloud top and bottom
    cloud_top = np.maximum(senkou_a_6h, senkou_b_6h)
    cloud_bottom = np.minimum(senkou_a_6h, senkou_b_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # start after warmup for Senkou Span B (52-period)
        # Skip if Ichimoku not ready
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or 
            np.isnan(senkou_b_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: price above cloud, Tenkan > Kijun, weekly uptrend
        if (close[i] > cloud_top[i] and 
            tenkan_6h[i] > kijun_6h[i] and 
            close[i] > senkou_b_1w_aligned[i]):
            if position != 1:
                position = 1
                signals[i] = 0.25
            else:
                signals[i] = 0.25
        # Short conditions: price below cloud, Tenkan < Kijun, weekly downtrend
        elif (close[i] < cloud_bottom[i] and 
              tenkan_6h[i] < kijun_6h[i] and 
              close[i] < senkou_b_1w_aligned[i]):
            if position != -1:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = -0.25
        # Exit conditions: opposite cloud cross
        elif position == 1 and close[i] < cloud_bottom[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > cloud_top[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals