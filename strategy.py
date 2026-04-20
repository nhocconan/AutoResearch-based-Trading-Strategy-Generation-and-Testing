#!/usr/bin/env python3
# 6h_1d_Ichimoku_Cloud_Breakout
# Hypothesis: Use 1d Ichimoku cloud as trend filter and 6x price for breakout entry. Long when price breaks above cloud with bullish TK cross; short when breaks below cloud with bearish TK cross. Works in bull/bear via cloud as dynamic support/resistance.
# Entry: Price breaks above/below cloud + TK cross in same direction. Exit: Price re-enters cloud or TK cross reverses.
# Uses 1d Tenkan/Kijun/Senkou spans. Targets 15-25 trades/year. Avoids whipsaw in sideways via cloud filter.

name = "6h_1d_Ichimoku_Cloud_Breakout"
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
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high_1d).rolling(window=9, min_periods=9).max()
    low_9 = pd.Series(low_1d).rolling(window=9, min_periods=9).min()
    tenkan = ((high_9 + low_9) / 2).values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high_1d).rolling(window=26, min_periods=26).max()
    low_26 = pd.Series(low_1d).rolling(window=26, min_periods=26).min()
    kijun = ((high_26 + low_26) / 2).values
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high_1d).rolling(window=52, min_periods=52).max()
    low_52 = pd.Series(low_1d).rolling(window=52, min_periods=52).min()
    senkou_b = ((high_52 + low_52) / 2)
    
    # Chikou Span (Lagging Span): close plotted 26 periods back
    # Not used for signals as it requires future data
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A and B)
        upper_cloud = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        lower_cloud = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # TK cross: Tenkan above/below Kijun
        tk_bullish = tenkan_aligned[i] > kijun_aligned[i]
        tk_bearish = tenkan_aligned[i] < kijun_aligned[i]
        
        if position == 0:
            # Long: price breaks above cloud + bullish TK cross
            if (close[i] > upper_cloud and tk_bullish):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below cloud + bearish TK cross
            elif (close[i] < lower_cloud and tk_bearish):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price re-enters cloud or TK cross turns bearish
            if close[i] < upper_cloud or not tk_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price re-enters cloud or TK cross turns bullish
            if close[i] > lower_cloud or not tk_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals