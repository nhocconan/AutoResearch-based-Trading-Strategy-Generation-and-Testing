#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud breakout with 1d TK Cross confirmation and volume filter.
- Uses Ichimoku (Tenkan, Kijun, Senkou Span A/B, Chikou) on 6h for structure
- 1d TK Cross (Tenkan/Kijun) as trend filter to avoid counter-trend trades
- Volume confirmation (2x 20-period MA) to reduce false breakouts
- Discrete position sizing (0.25) to minimize fee churn
- Works in bull/bear via trend filter: only long when 1d TK Cross bullish, short when bearish
- Target: 12-30 trades/year per symbol (50-120 total over 4 years)
"""

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
    volume = prices['volume'].values
    
    # Calculate Ichimoku on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    highest_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    lowest_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (highest_tenkan + lowest_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    highest_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    lowest_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (highest_kijun + lowest_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    highest_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    lowest_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((highest_senkou_b + lowest_senkou_b) / 2)
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    chikou = close  # Will be aligned later with shift
    
    # Calculate 1d TK Cross for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < period_kijun:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Tenkan and Kijun
    highest_tenkan_1d = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    lowest_tenkan_1d = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_1d = (highest_tenkan_1d + lowest_tenkan_1d) / 2
    
    highest_kijun_1d = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    lowest_kijun_1d = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_1d = (highest_kijun_1d + lowest_kijun_1d) / 2
    
    # 1d TK Cross: bullish when Tenkan > Kijun, bearish when Tenkan < Kijun
    tk_bullish_1d = tenkan_1d > kijun_1d
    tk_bearish_1d = tenkan_1d < kijun_1d
    
    # Align all 6h Ichimoku components
    tenkan_aligned = align_htf_to_ltf(prices, prices, tenkan)  # same timeframe, no shift needed
    kijun_aligned = align_htf_to_ltf(prices, prices, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, prices, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, prices, senkou_b)
    chikou_aligned = align_htf_to_ltf(prices, prices, chikou)
    
    # Align 1d TK Cross
    tk_bullish_aligned = align_htf_to_ltf(prices, df_1d, tk_bullish_1d.astype(float))
    tk_bearish_aligned = align_htf_to_ltf(prices, df_1d, tk_bearish_1d.astype(float))
    
    # Volume confirmation: 2x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    # Need: 52 for Senkou B, 26 for others, 20 for volume MA
    start_idx = max(52 + 26, 20)  # Senkou B is shifted 26 ahead, so need 52+26 to see it
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(chikou_aligned[i]) or np.isnan(tk_bullish_aligned[i]) or
            np.isnan(tk_bearish_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Ichimoku Cloud: bullish when price > Senkou Span A and Senkou Span A > Senkou Span B
        # Bearish when price < Senkou Span A and Senkou Span A < Senkou Span B
        bullish_cloud = (close[i] > senkou_a_aligned[i]) and (senkou_a_aligned[i] > senkou_b_aligned[i])
        bearish_cloud = (close[i] < senkou_a_aligned[i]) and (senkou_a_aligned[i] < senkou_b_aligned[i])
        
        # TK Cross on 6h: Tenkan > Kijun = bullish, Tenkan < Kijun = bearish
        tk_bullish_6h = tenkan_aligned[i] > kijun_aligned[i]
        tk_bearish_6h = tenkan_aligned[i] < kijun_aligned[i]
        
        # Volume filter
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Bullish cloud AND bullish TK (6h) AND bullish 1d TK Cross AND volume
            if bullish_cloud and tk_bullish_6h and tk_bullish_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Bearish cloud AND bearish TK (6h) AND bearish 1d TK Cross AND volume
            elif bearish_cloud and tk_bearish_6h and tk_bearish_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: opposite cloud or TK cross reversal
            exit_signal = False
            if position == 1:
                # Exit long on bearish cloud or bearish TK cross
                if not bullish_cloud or tk_bearish_6h:
                    exit_signal = True
            elif position == -1:
                # Exit short on bullish cloud or bullish TK cross
                if not bearish_cloud or tk_bullish_6h:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Ichimoku_Cloud_TKCross_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0