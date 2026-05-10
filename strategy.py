#!/usr/bin/env python3
"""
6H_Ichimoku_Cloud_1dTrend_PriceAction
Hypothesis: Ichimoku cloud (TK cross + cloud color) from 1d timeframe combined with 6h price action (close > open for long, close < open for short) captures trend continuation with low trade frequency. Uses only price action and 1d Ichimoku components to avoid overfitting. Designed for 6-12 trades/year per symbol to minimize fee drag while capturing major trend moves in both bull and bear markets.
"""

name = "6H_Ichimoku_Cloud_1dTrend_PriceAction"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_price = prices['open'].values
    
    # 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 periods for Ichimoku
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components (standard settings: 9, 26, 52)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Warmup for Ichimoku (52 periods)
    
    for i in range(start_idx, n):
        # Skip if any Ichimoku component is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud color and TK cross
        # Green cloud: Senkou A > Senkou B (bullish)
        # Red cloud: Senkou A < Senkou B (bearish)
        # TK cross: Tenkan > Kijun (bullish), Tenkan < Kijun (bearish)
        bullish_cloud = senkou_a_aligned[i] > senkou_b_aligned[i]
        bearish_cloud = senkou_a_aligned[i] < senkou_b_aligned[i]
        tk_bullish = tenkan_aligned[i] > kijun_aligned[i]
        tk_bearish = tenkan_aligned[i] < kijun_aligned[i]
        
        # Price action: bullish/bearish candle
        bullish_candle = close[i] > open_price[i]
        bearish_candle = close[i] < open_price[i]
        
        if position == 0:
            # Long entry: Price above cloud + TK bullish cross + bullish candle
            if (close[i] > senkou_a_aligned[i] and close[i] > senkou_b_aligned[i] and
                tk_bullish and bullish_candle):
                signals[i] = 0.25
                position = 1
            # Short entry: Price below cloud + TK bearish cross + bearish candle
            elif (close[i] < senkou_a_aligned[i] and close[i] < senkou_b_aligned[i] and
                  tk_bearish and bearish_candle):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price closes below cloud OR TK bearish cross
            if (close[i] < senkou_a_aligned[i] or close[i] < senkou_b_aligned[i] or
                tk_bearish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price closes above cloud OR TK bullish cross
            if (close[i] > senkou_a_aligned[i] or close[i] > senkou_b_aligned[i] or
                tk_bullish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals