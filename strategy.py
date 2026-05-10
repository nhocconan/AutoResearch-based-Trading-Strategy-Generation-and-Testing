#!/usr/bin/env python3
# 6h_Ichimoku_Trend_Follow
# Hypothesis: Ichimoku system (Tenkan/Kijun cross + Cloud) on 1d provides trend direction,
# while 6h Tenkan/Kijun cross provides entry timing. Works in bull/bear by following
# the higher timeframe trend. Uses volume confirmation to avoid false breakouts.
# Target: 15-35 trades/year (60-140 total over 4 years).

name = "6h_Ichimoku_Trend_Follow"
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
    
    # Get daily data for Ichimoku trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on daily data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun_1d = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    high_52 = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (high_52 + low_52) / 2
    
    # Align Ichimoku components to 6h timeframe (wait for daily close)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Calculate Ichimoku components on 6h for entry signals
    high_9_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6h = (high_9_6h + low_9_6h) / 2
    
    high_26_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6h = (high_26_6h + low_26_6h) / 2
    
    # Volume confirmation (24-period MA on 6h chart)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 52 periods for Senkou B
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine if price is above or below the cloud
        cloud_top = np.maximum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        cloud_bottom = np.minimum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # 1d trend: Tenkan/Kijun cross
        bullish_trend_1d = tenkan_1d_aligned[i] > kijun_1d_aligned[i]
        bearish_trend_1d = tenkan_1d_aligned[i] < kijun_1d_aligned[i]
        
        # 6h entry signal: Tenkan/Kijun cross
        bullish_cross_6h = tenkan_6h[i] > kijun_6h[i]
        bearish_cross_6h = tenkan_6h[i] < kijun_6h[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: 1d bullish trend + price above cloud + 6h bullish cross + volume
            if bullish_trend_1d and price_above_cloud and bullish_cross_6h and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: 1d bearish trend + price below cloud + 6h bearish cross + volume
            elif bearish_trend_1d and price_below_cloud and bearish_cross_6h and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend turns bearish or price drops below cloud
            if not bullish_trend_1d or not price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend turns bullish or price rises above cloud
            if not bearish_trend_1d or not price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals