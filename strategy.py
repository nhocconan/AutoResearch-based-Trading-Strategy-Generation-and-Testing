#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_With_1dTrend
Hypothesis: Ichimoku cloud breakout with 1d trend filter. Enter long when price breaks above cloud in bullish 1d regime, short when breaks below cloud in bearish 1d regime. Uses 6h timeframe for balance between signal quality and trade frequency. Weekly trend filter avoids counter-trend trades. Designed to work in both bull and bear markets by following higher timeframe trend.
"""

name = "6h_Ichimoku_Cloud_Breakout_With_1dTrend"
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
    
    # Get 1d data for trend filter and Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Get weekly data for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 26:
        return np.zeros(n)
    
    # Ichimoku components on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    max_high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period_senkou = 52
    max_high_senkou = pd.Series(high_1d).rolling(window=period_senkou, min_periods=period_senkou).max().values
    min_low_senkou = pd.Series(low_1d).rolling(window=period_senkou, min_periods=period_senkou).min().values
    senkou_span_b = (max_high_senkou + min_low_senkou) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    cloud_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    # 1d trend filter: price vs Kumo (cloud)
    # Bullish when price above cloud, bearish when price below cloud
    price_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
    trend_1d_bullish = price_1d_aligned > cloud_top
    trend_1d_bearish = price_1d_aligned < cloud_bottom
    
    # Weekly trend filter: avoid counter-trend trades
    close_1w = df_1w['close'].values
    ema_26_1w = pd.Series(close_1w).ewm(span=26, adjust=False, min_periods=26).mean().values
    ema_26_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_26_1w)
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    weekly_uptrend = close_1w_aligned > ema_26_1w_aligned
    weekly_downtrend = close_1w_aligned < ema_26_1w_aligned
    
    # Volume confirmation: 24-period average on 6h
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = np.divide(volume, vol_ma24, out=np.zeros_like(volume), where=vol_ma24!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 26, 24)  # Warmup for Ichimoku and volume
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(price_1d_aligned[i]) or np.isnan(ema_26_1w_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Cloud breakout conditions
        price_above_cloud = close[i] > cloud_top[i]
        price_below_cloud = close[i] < cloud_bottom[i]
        
        if position == 0:
            # Long: price breaks above cloud in bullish 1d regime with weekly uptrend and volume
            if (price_above_cloud and 
                trend_1d_bullish[i] and 
                weekly_uptrend[i] and 
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below cloud in bearish 1d regime with weekly downtrend and volume
            elif (price_below_cloud and 
                  trend_1d_bearish[i] and 
                  weekly_downtrend[i] and 
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below cloud or weekly trend turns bearish
            if (close[i] < cloud_bottom[i] or not weekly_uptrend[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above cloud or weekly trend turns bullish
            if (close[i] > cloud_top[i] or not weekly_downtrend[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals