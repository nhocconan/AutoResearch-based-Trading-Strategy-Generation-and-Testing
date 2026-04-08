#!/usr/bin/env python3
# 6h_1d_ichimoku_cloud_trend_v1
# Hypothesis: 6h Ichimoku cloud strategy with 1d trend filter for BTC/ETH/SOL.
# Long: price > 6h cloud AND Tenkan > Kijun (bullish momentum) AND 1d close > 1d Senkou Span A (bullish regime)
# Short: price < 6h cloud AND Tenkan < Kijun (bearish momentum) AND 1d close < 1d Senkou Span A (bearish regime)
# Exit: price crosses opposite 6h cloud boundary (Tenkan/Kijun average) OR momentum reverses
# Uses 6h primary timeframe with 1d HTF for regime filter.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ichimoku_cloud_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 6h Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2 shifted 26 periods ahead
    period_senkou_b = 52
    high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((high_senkou_b + low_senkou_b) / 2)
    
    # Get 1d data for regime filter (1d Ichimoku cloud)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Ichimoku components for regime filter
    high_tenkan_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_tenkan_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (high_tenkan_1d + low_tenkan_1d) / 2
    
    high_kijun_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_kijun_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (high_kijun_1d + low_kijun_1d) / 2
    
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    
    high_senkou_b_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_senkou_b_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = ((high_senkou_b_1d + low_senkou_b_1d) / 2)
    
    # Align 1d indicators to 6h timeframe
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required values are NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        price = close[i]
        
        # 6h Ichimoku cloud boundaries (using Senkou Span A and B)
        cloud_top = max(senkou_a[i], senkou_b[i])
        cloud_bottom = min(senkou_a[i], senkou_b[i])
        
        # 6h momentum: Tenkan vs Kijun
        tenkan_above_kijun = tenkan[i] > kijun[i]
        tenkan_below_kijun = tenkan[i] < kijun[i]
        
        # 1d regime filter: price vs 1d cloud
        price_1d = close_1d[min(i // 4, len(close_1d)-1)]  # Approximate 1d close for regime
        # Better: use the actual aligned 1d close from HTF data
        # We'll use the 1d close price aligned to 6f for regime check
        
        # Get aligned 1d close for regime (we need to extract it from df_1d)
        # Since we don't have it pre-aligned, we'll use Senkou spans as proxy for regime
        # Bullish regime: 1d Senkou Span A > Senkou Span B (cloud bullish)
        # Bearish regime: 1d Senkou Span A < Senkou Span B (cloud bearish)
        cloud_bullish_1d = senkou_a_1d_aligned[i] > senkou_b_1d_aligned[i]
        cloud_bearish_1d = senkou_a_1d_aligned[i] < senkou_b_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below cloud OR momentum turns bearish
            if price < cloud_bottom or not tenkan_above_kijun:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above cloud OR momentum turns bullish
            if price > cloud_top or not tenkan_below_kijun:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: price above 6h cloud AND bullish momentum AND bullish 1d regime
            if price > cloud_top and tenkan_above_kijun and cloud_bullish_1d:
                position = 1
                signals[i] = 0.25
            # Short entry: price below 6h cloud AND bearish momentum AND bearish 1d regime
            elif price < cloud_bottom and tenkan_below_kijun and cloud_bearish_1d:
                position = -1
                signals[i] = -0.25
    
    return signals