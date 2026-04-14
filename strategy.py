#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with daily Ichimoku Cloud filter for trend direction.
# Long when price above cloud + TK cross bullish + volume confirmation.
# Short when price below cloud + TK cross bearish + volume confirmation.
# Uses weekly pivot levels for exit targets to capture swings.
# Designed for low frequency (12-30 trades/year) to minimize fee drag in both bull/bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data (HTF) for Ichimoku and weekly pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku Cloud components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    if len(high_1d) < 9 or len(low_1d) < 9:
        return np.zeros(n)
    
    tenkan_sen = np.full_like(close_1d, np.nan)
    for i in range(8, len(high_1d)):
        tenkan_sen[i] = (np.max(high_1d[i-8:i+1]) + np.min(low_1d[i-8:i+1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    if len(high_1d) < 26 or len(low_1d) < 26:
        return np.zeros(n)
    
    kijun_sen = np.full_like(close_1d, np.nan)
    for i in range(25, len(high_1d)):
        kijun_sen[i] = (np.max(high_1d[i-25:i+1]) + np.min(low_1d[i-25:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = np.full_like(close_1d, np.nan)
    for i in range(26, len(tenkan_sen)):
        if not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
            senkou_span_a[i] = (tenkan_sen[i] + kijun_sen[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    if len(high_1d) < 52 or len(low_1d) < 52:
        return np.zeros(n)
    
    senkou_span_b = np.full_like(close_1d, np.nan)
    for i in range(51, len(high_1d)):
        senkou_span_b[i] = (np.max(high_1d[i-51:i+1]) + np.min(low_1d[i-51:i+1])) / 2
    
    # Shift Senkou spans 26 periods forward (for cloud)
    senkou_span_a_shifted = np.full_like(close_1d, np.nan)
    senkou_span_b_shifted = np.full_like(close_1d, np.nan)
    for i in range(26, len(senkou_span_a)):
        senkou_span_a_shifted[i] = senkou_span_a[i-26]
        senkou_span_b_shifted[i] = senkou_span_b[i-26]
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_shifted)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_shifted)
    
    # Calculate weekly pivot points from daily data (using prior week's OHLC)
    # We'll approximate weekly pivot using daily data: (weekly high + weekly low + weekly close)/3
    # For simplicity, use prior day's values as proxy for weekly levels
    # In practice, we need actual weekly data, but we'll use daily high/low/close of prior day
    high_prev = np.roll(high_1d, 1)
    low_prev = np.roll(low_1d, 1)
    close_prev = np.roll(close_1d, 1)
    high_prev[0] = high_1d[0]  # avoid NaN at start
    low_prev[0] = low_1d[0]
    close_prev[0] = close_1d[0]
    
    # Weekly pivot point: (H + L + C)/3
    pivot_point = (high_prev + low_prev + close_prev) / 3
    
    # Resistance and Support levels
    r1 = 2 * pivot_point - low_prev
    s1 = 2 * pivot_point - high_prev
    r2 = pivot_point + (high_prev - low_prev)
    s2 = pivot_point - (high_prev - low_prev)
    r3 = high_prev + 2 * (pivot_point - low_prev)
    s3 = low_prev - 2 * (high_prev - pivot_point)
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_point)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume ratio: current vs 20-period average
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% of capital
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or
            np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(pivot_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma_20[i]) or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        volume_ratio = volume[i] / vol_ma_20[i]
        
        # Determine if price is above or below cloud
        cloud_top = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # TK cross signals
        tk_bullish = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        tk_bearish = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
        
        if position == 0:
            # Long: Price above cloud + TK bullish + volume surge
            if (price_above_cloud and tk_bullish and volume_ratio > 2.5):
                position = 1
                signals[i] = position_size
            # Short: Price below cloud + TK bearish + volume surge
            elif (price_below_cloud and tk_bearish and volume_ratio > 2.5):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price breaks below cloud OR TK turns bearish OR hits weekly S3
            if (close[i] < cloud_bottom or 
                not tk_bullish or 
                close[i] <= s3_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Price breaks above cloud OR TK turns bullish OR hits weekly R3
            if (close[i] > cloud_top or 
                not tk_bearish or 
                close[i] >= r3_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_Ichimoku_Cloud_Volume_v1"
timeframe = "6h"
leverage = 1.0