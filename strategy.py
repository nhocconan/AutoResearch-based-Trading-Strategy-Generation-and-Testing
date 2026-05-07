#!/usr/bin/env python3
name = "6h_Ichimoku_TK_Cross_Cloud_Filter"
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
    
    # Get daily data for Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Ichimoku components (using 1d data)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (high_9 + low_9) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun_sen = (high_26 + low_26) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (high_52 + low_52) / 2.0
    
    # Align Ichimoku components to 6h timeframe (available after daily bar closes)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Wait for Senkou Span B calculation
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or 
            np.isnan(senkou_b_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud color (green = bullish, red = bearish)
        # Cloud is between Senkou Span A and B
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # TK Cross signals
        tk_cross_bullish = tenkan_aligned[i] > kijun_aligned[i]
        tk_cross_bearish = tenkan_aligned[i] < kijun_aligned[i]
        
        # Price relative to cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        if position == 0:
            # Long: TK cross bullish + price above cloud (bullish cloud)
            if tk_cross_bullish and price_above_cloud and (senkou_a_aligned[i] > senkou_b_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: TK cross bearish + price below cloud (bearish cloud)
            elif tk_cross_bearish and price_below_cloud and (senkou_a_aligned[i] < senkou_b_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TK cross bearish OR price drops below cloud
            if tk_cross_bearish or close[i] < cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TK cross bullish OR price rises above cloud
            if tk_cross_bullish or close[i] > cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: On 6h timeframe, Ichimoku TK (Tenkan/Kijun) cross combined with cloud filter from daily timeframe provides high-probability entries. The daily Ichimoku cloud acts as dynamic support/resistance, while the TK cross signals momentum shifts. In bull markets, longs are taken when TK crosses bullish above a bullish cloud (Senkou A > Senkou B). In bear markets, shorts are taken when TK crosses bearish below a bearish cloud (Senkou A < Senkou B). This approach avoids whipsaws by requiring alignment with the higher timeframe Ichimoku structure. Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while capturing significant trends.