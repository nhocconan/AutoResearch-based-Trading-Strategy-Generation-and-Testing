#!/usr/bin/env python3
# 6h_Ichimoku_TK_Cross_Cloud_Filter_1d
# Hypothesis: Ichimoku TK cross with cloud filter from 1d timeframe provides high-probability entries in trending markets.
# Long when: TK cross bullish (Tenkan > Kijun) AND price above 1d cloud (Senkou Span A/B).
# Short when: TK cross bearish (Tenkan < Kijun) AND price below 1d cloud.
# Uses volume confirmation to avoid low-conviction signals.
# Works in bull markets (follows uptrends) and bear markets (follows downtrends via short signals).
# Target: 50-150 total trades over 4 years.

name = "6h_Ichimoku_TK_Cross_Cloud_Filter_1d"
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
    
    # Get daily data for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 6h chart
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max()
    min_low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min()
    tenkan = ((max_high_tenkan + min_low_tenkan) / 2).values
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    max_high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max()
    min_low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min()
    kijun = ((max_high_kijun + min_low_kijun) / 2).values
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max()
    min_low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()
    senkou_b = ((max_high_senkou_b + min_low_senkou_b) / 2)
    # Shift both Senkou spans forward by 26 periods (will align via align_htf_to_ltf later)
    
    # Calculate Ichimoku components on 1d chart for cloud filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Tenkan-sen 1d
    tenkan_1d = ((pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2).values
    # Kijun-sen 1d
    kijun_1d = ((pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2).values
    # Senkou Span A 1d
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    # Senkou Span B 1d
    senkou_b_1d = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                    pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2)
    
    # Align 1d Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d, additional_delay_bars=26)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d, additional_delay_bars=26)
    
    # Volume confirmation (20-period MA on 6h chart)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Tenkan (9), Kijun (26), Senkou B (52), volume MA (20)
    start_idx = max(9, 26, 52, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # TK cross on 6h chart
        tk_cross_bullish = tenkan[i] > kijun[i]
        tk_cross_bearish = tenkan[i] < kijun[i]
        
        # Price relative to 1d cloud
        # Cloud top = max(Senkou A, Senkou B), Cloud bottom = min(Senkou A, Senkou B)
        cloud_top = np.maximum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        cloud_bottom = np.minimum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: bullish TK cross + price above cloud + volume
            if tk_cross_bullish and price_above_cloud and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish TK cross + price below cloud + volume
            elif tk_cross_bearish and price_below_cloud and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish TK cross or price drops below cloud
            if tk_cross_bearish or not price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish TK cross or price rises above cloud
            if tk_cross_bullish or not price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals