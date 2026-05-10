#!/usr/bin/env python3
# 6H_Ichimoku_Cloud_Trend_Follow
# Hypothesis: Ichimoku cloud from daily timeframe acts as dynamic support/resistance.
# In bull markets: price above cloud = bullish, enter long on TK cross up.
# In bear markets: price below cloud = bearish, enter short on TK cross down.
# Cloud filters out false signals during sideways markets.
# Target: 15-25 trades/year per symbol.

name = "6H_Ichimoku_Cloud_Trend_Follow"
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
    
    # 6h indicators
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    tenkan_sen = ((high_s.rolling(window=period_tenkan, min_periods=period_tenkan).max() + 
                   low_s.rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2).values
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    kijun_sen = ((high_s.rolling(window=period_kijun, min_periods=period_kijun).max() + 
                  low_s.rolling(window=period_kijun, min_periods=period_kijun).min()) / 2).values
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    # We'll handle the shift in alignment
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    senkou_span_b = ((high_s.rolling(window=period_senkou_b, min_periods=period_senkou_b).max() + 
                      low_s.rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    # Not used for entry but can be used for confirmation
    
    # Daily Ichimoku cloud (for trend filtering)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    high_1d_s = pd.Series(high_1d)
    low_1d_s = pd.Series(low_1d)
    close_1d_s = pd.Series(close_1d)
    
    # Daily Tenkan-sen (9-period)
    tenkan_1d = ((high_1d_s.rolling(window=9, min_periods=9).max() + 
                  low_1d_s.rolling(window=9, min_periods=9).min()) / 2).values
    
    # Daily Kijun-sen (26-period)
    kijun_1d = ((high_1d_s.rolling(window=26, min_periods=26).max() + 
                 low_1d_s.rolling(window=26, min_periods=26).min()) / 2).values
    
    # Daily Senkou Span A
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    
    # Daily Senkou Span B (52-period)
    senkou_b_1d = ((high_1d_s.rolling(window=52, min_periods=52).max() + 
                    low_1d_s.rolling(window=52, min_periods=52).min()) / 2)
    
    # Align daily Ichimoku components to 6h
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Align 6h Ichimoku components (no shift needed as we use current values)
    # Note: For actual Ichimoku, Senkou spans are shifted, but we use current values
    # for cloud calculation and rely on alignment for proper timing
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate current cloud boundaries (using aligned daily values)
        top_cloud = max(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        bottom_cloud = min(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        # TK cross signals
        tk_cross_up = tenkan_sen[i] > kijun_sen[i] and tenkan_sen[i-1] <= kijun_sen[i-1]
        tk_cross_down = tenkan_sen[i] < kijun_sen[i] and tenkan_sen[i-1] >= kijun_sen[i-1]
        
        # Price position relative to cloud
        price_above_cloud = close[i] > top_cloud
        price_below_cloud = close[i] < bottom_cloud
        
        if position == 0:
            # Enter long: price above cloud + TK cross up
            if price_above_cloud and tk_cross_up:
                signals[i] = 0.25
                position = 1
            # Enter short: price below cloud + TK cross down
            elif price_below_cloud and tk_cross_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below cloud or TK cross down
            if price_below_cloud or tk_cross_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above cloud or TK cross up
            if price_above_cloud or tk_cross_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals