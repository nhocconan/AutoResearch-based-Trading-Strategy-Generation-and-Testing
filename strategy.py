#!/usr/bin/env python3
# 1D_1W_Ichimoku_Trend_Follow
# Hypothesis: Ichimoku Cloud on weekly timeframe defines trend, with entry on daily retracement to Kijun-sen.
# Long when: weekly Senkou Span A > Senkou Span B (bullish cloud), price above weekly Kijun-sen, and daily close crosses above daily Kijun-sen.
# Short when: weekly Senkou Span A < Senkou Span B (bearish cloud), price below weekly Kijun-sen, and daily close crosses below daily Kijun-sen.
# Uses weekly cloud for trend filter and weekly/daily Kijun-sen for entries.
# Works in bull/bear by following weekly trend direction. Target: 15-25 trades/year per symbol.

name = "1D_1W_Ichimoku_Trend_Follow"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get weekly data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    high_9 = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    high_26 = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_sen = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    high_52 = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_52 + low_52) / 2)
    
    # Chikou Span (Lagging Span): current close shifted 26 periods back (not used for signals)
    
    # Bullish cloud: Senkou Span A > Senkou Span B
    bullish_cloud = senkou_a > senkou_b
    bearish_cloud = senkou_a < senkou_b
    
    # Align weekly indicators to daily
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen)
    bullish_cloud_aligned = align_htf_to_ltf(prices, df_1w, bullish_cloud.astype(float))
    bearish_cloud_aligned = align_htf_to_ltf(prices, df_1w, bearish_cloud.astype(float))
    
    # Get daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily Kijun-sen (Base Line): (26-period high + low)/2
    high_26d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen_1d = (high_26d + low_26d) / 2
    
    # Align daily Kijun-sen to itself (no shift needed, but for consistency)
    kijun_sen_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kijun_sen_aligned[i]) or
            np.isnan(bullish_cloud_aligned[i]) or np.isnan(bearish_cloud_aligned[i]) or
            np.isnan(kijun_sen_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bullish_cloud = bullish_cloud_aligned[i] > 0.5
        bearish_cloud = bearish_cloud_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: bullish cloud + price above weekly Kijun-sen + daily close crosses above daily Kijun-sen
            if bullish_cloud and close[i] > kijun_sen_aligned[i] and close_1d[i] > kijun_sen_1d_aligned[i]:
                # Check for crossover: previous close below, current close above
                if i > 0 and close_1d[i-1] <= kijun_sen_1d_aligned[i-1]:
                    signals[i] = 0.25
                    position = 1
            # Enter short: bearish cloud + price below weekly Kijun-sen + daily close crosses below daily Kijun-sen
            elif bearish_cloud and close[i] < kijun_sen_aligned[i] and close_1d[i] < kijun_sen_1d_aligned[i]:
                # Check for crossover: previous close above, current close below
                if i > 0 and close_1d[i-1] >= kijun_sen_1d_aligned[i-1]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: bearish cloud or price crosses below weekly Kijun-sen
            if bearish_cloud or close[i] < kijun_sen_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish cloud or price crosses above weekly Kijun-sen
            if bullish_cloud or close[i] > kijun_sen_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals