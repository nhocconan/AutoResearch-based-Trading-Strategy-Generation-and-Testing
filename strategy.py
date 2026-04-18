#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_TK_Cross_WeeklyTrend
6h strategy using Ichimoku cloud with Tenkan-Kijun cross and weekly trend filter.
- Long: Price above Ichimoku cloud + TK cross bullish + weekly trend up
- Short: Price below Ichimoku cloud + TK cross bearish + weekly trend down
- Exit: Opposite conditions
Ichimoku provides dynamic support/resistance and trend strength.
Works in both bull and bear markets by following higher timeframe trend.
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
    
    # Weekly EMA25 for trend filter
    ema_25_1w = pd.Series(close_1w).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema_25_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_25_1w)
    
    # Ichimoku components (9, 26, 52)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # Align Ichimoku components to current timeframe
    tenkan_aligned = align_htf_to_ltf(prices, prices, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, prices, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, prices, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, prices, senkou_b)
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # need enough for Senkou B
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(ema_25_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend
        weekly_up = close[i] > ema_25_1w_aligned[i]
        weekly_down = close[i] < ema_25_1w_aligned[i]
        
        # Ichimoku conditions
        price_above_cloud = close[i] > cloud_top[i]
        price_below_cloud = close[i] < cloud_bottom[i]
        tk_bullish = tenkan_aligned[i] > kijun_aligned[i]
        tk_bearish = tenkan_aligned[i] < kijun_aligned[i]
        
        if position == 0:
            # Long: price above cloud + TK bullish + weekly up
            if price_above_cloud and tk_bullish and weekly_up:
                signals[i] = 0.25
                position = 1
            # Short: price below cloud + TK bearish + weekly down
            elif price_below_cloud and tk_bearish and weekly_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below cloud or TK bearish or weekly down
            if price_below_cloud or not tk_bullish or not weekly_up:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above cloud or TK bullish or weekly up
            if price_above_cloud or tk_bullish or weekly_up:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_WeeklyTrend"
timeframe = "6h"
leverage = 1.0