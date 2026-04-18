#!/usr/bin/env python3
"""
1d_Ichimoku_Cloud_Trend_1wTrend_Filter
Hypothesis: Daily Ichimoku Cloud breakouts with weekly trend filter to capture major trends in both bull and bear markets.
Uses cloud as dynamic support/resistance, TK cross for momentum, and weekly trend to avoid counter-trend trades.
Designed for low trade frequency (target: 10-25/year) with strong performance in trending markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_52 + low_52) / 2)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA25 for trend filter
    ema25_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 25:
        ema25_1w[24] = np.mean(close_1w[0:25])
        alpha = 2 / (25 + 1)
        for i in range(25, len(close_1w)):
            ema25_1w[i] = close_1w[i] * alpha + ema25_1w[i-1] * (1 - alpha)
    
    # Align weekly EMA25 to daily timeframe
    ema25_1w_aligned = align_htf_to_ltf(prices, df_1w, ema25_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52 + 26  # Senkou spans need 52 periods + 26 shift
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(ema25_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine if price is above or below cloud
        cloud_top = max(senkou_a[i], senkou_b[i])
        cloud_bottom = min(senkou_a[i], senkou_b[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # TK Cross signals
        tk_cross_up = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
        tk_cross_down = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
        
        if position == 0:
            # Long: price above cloud + TK cross up + weekly uptrend
            if (price_above_cloud and tk_cross_up and 
                close[i] > ema25_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below cloud + TK cross down + weekly downtrend
            elif (price_below_cloud and tk_cross_down and 
                  close[i] < ema25_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below cloud or TK cross down
            if price_below_cloud or tk_cross_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above cloud or TK cross up
            if price_above_cloud or tk_cross_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Ichimoku_Cloud_Trend_1wTrend_Filter"
timeframe = "1d"
leverage = 1.0