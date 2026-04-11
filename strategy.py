#!/usr/bin/env python3
# 6h_12h_1d_ichimoku_cloud_v1
# Strategy: 6h Ichimoku Cloud with 1d trend filter and volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Ichimoku Cloud provides strong support/resistance and trend direction.
# In bull markets: price above cloud (Senkou Span A/B) with TK cross bullish.
# In bear markets: price below cloud with TK cross bearish.
# Uses 1d ADX > 25 to filter for trending markets only, reducing whipsaws in ranging conditions.
# Volume confirmation ensures breaks have conviction.
# Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_ichimoku_cloud_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d ADX(14) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX components
    plus_dm = np.zeros(len(high_1d))
    minus_dm = np.zeros(len(high_1d))
    tr = np.zeros(len(high_1d))
    
    for i in range(1, len(high_1d)):
        high_diff = high_1d[i] - high_1d[i-1]
        low_diff = low_1d[i-1] - low_1d[i]
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        tr[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    # Smooth with Wilder's smoothing (equivalent to RMA)
    atr = np.zeros(len(high_1d))
    plus_dm_smooth = np.zeros(len(high_1d))
    minus_dm_smooth = np.zeros(len(high_1d))
    
    # Initial values
    atr[0] = tr[0]
    plus_dm_smooth[0] = plus_dm[0]
    minus_dm_smooth[0] = minus_dm[0]
    
    # Wilder's smoothing: new_val = (prev * (n-1) + current) / n
    for i in range(1, len(high_1d)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
        plus_dm_smooth[i] = (plus_dm_smooth[i-1] * 13 + plus_dm[i]) / 14
        minus_dm_smooth[i] = (minus_dm_smooth[i-1] * 13 + minus_dm[i]) / 14
    
    # Avoid division by zero
    plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0)
    minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    
    # ADX: smoothed DX
    adx = np.zeros(len(dx))
    adx[0] = dx[0]
    for i in range(1, len(dx)):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    adx_1d = adx
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Ichimoku Cloud calculation (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).max().values
    
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).min().values
    
    tenkan_sen = (rolling_max(high, 9) + rolling_min(low, 9)) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun_sen = (rolling_max(high, 26) + rolling_min(low, 26)) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    senkou_span_b = (rolling_max(high, 52) + rolling_min(low, 52)) / 2
    
    # Align Ichimoku components to 6t timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Need enough data for Ichimoku
        # Skip if any required data is invalid
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or np.isnan(senkou_span_a_aligned[i]) or 
            np.isnan(senkou_span_b_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: only trade in trending markets (ADX > 25)
        trending = adx_1d_aligned[i] > 25
        
        # Determine cloud boundaries (account for Senkou Span B being plotted 26 periods ahead)
        # For current price, Senkou Span A/B values from 26 periods ago form the cloud
        if i >= 26:
            span_a = senkou_span_a_aligned[i-26]
            span_b = senkou_span_b_aligned[i-26]
        else:
            # Not enough data for cloud, skip
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Cloud top and bottom
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        # Price above/below cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # TK Cross
        tk_bullish = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        tk_bearish = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
        
        # Entry logic: TK cross + price vs cloud + trend + volume
        if (tk_bullish and price_above_cloud and trending and vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        elif (tk_bearish and price_below_cloud and trending and vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: TK cross reversal or price enters cloud
        elif position == 1 and (tk_bearish or not price_above_cloud):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (tk_bullish or not price_below_cloud):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals