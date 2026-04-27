#!/usr/bin/env python3
"""
Hypothesis: 6-hour Ichimoku Cloud breakout with daily trend filter and volume confirmation.
Uses Ichimoku (Tenkan-sen, Kijun-sen, Senkou Span A/B) to detect trend and momentum.
Long when price breaks above Kumo (cloud) with bullish TK cross and daily uptrend.
Short when price breaks below Kumo with bearish TK cross and daily downtrend.
Volume filter ensures breakouts have conviction.
Designed to work in both bull and bear markets by using daily trend as filter.
Target: 15-35 trades/year per symbol (60-140 total over 4 years) to minimize fee drag.
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
    
    # Get daily data for Ichimoku and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                      pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2).shift(26)
    
    # Align Ichimoku components to 1d timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    # Get daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 12-hour data for volume filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12-hour volume MA(20)
    vol_12h = df_12h['volume'].values
    vol_ma_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Ichimoku (52+26=78), volume MA, and daily EMA
    start_idx = max(78, 20, 50)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or 
            np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(vol_ma_20_12h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Ichimoku values
        tenkan = tenkan_sen_aligned[i]
        kijun = kijun_sen_aligned[i]
        span_a = senkou_span_a_aligned[i]
        span_b = senkou_span_b_aligned[i]
        
        # Cloud boundaries (Senkou Span A/B)
        top_cloud = max(span_a, span_b)
        bottom_cloud = min(span_a, span_b)
        
        # Volume filter: volume > 1.5x 12h average
        vol_now = volume[i]
        vol_ma = vol_ma_20_12h_aligned[i]
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Trend filter: daily EMA(50)
        trend_1d = ema_50_1d_aligned[i]
        
        # TK Cross: Tenkan-sen / Kijun-sen relationship
        tk_bullish = tenkan > kijun
        tk_bearish = tenkan < kijun
        
        # Entry conditions
        if position == 0:
            # Long: Price breaks above cloud + bullish TK cross + volume + daily uptrend
            if (close[i] > top_cloud and tk_bullish and 
                vol_filter and close[i] > trend_1d):
                signals[i] = size
                position = 1
            # Short: Price breaks below cloud + bearish TK cross + volume + daily downtrend
            elif (close[i] < bottom_cloud and tk_bearish and 
                  vol_filter and close[i] < trend_1d):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price returns to cloud or bearish TK cross
            if (close[i] < top_cloud or not tk_bullish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Price returns to cloud or bullish TK cross
            if (close[i] > bottom_cloud or tk_bullish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_IchimokuCloudBreakout_DailyTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0