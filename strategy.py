#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation
# Uses Ichimoku (Tenkan/Kijun cross) on 6h for entry timing, filtered by 1d price vs Cloud (Kumo) for trend direction
# Volume spike confirms institutional participation. Only trades in strong trends (price above/below cloud).
# Works in bull/bear markets by requiring price to be on correct side of cloud.
# Targets 15-25 trades/year (~60-100 total over 4 years) to minimize fee drag.

name = "6h_Ichimoku_1dCloud_Filter_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku components on 6h (9, 26, 52 periods)
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = np.full_like(high, np.nan)
    for i in range(tenkan_period - 1, n):
        tenkan_sen[i] = (np.max(high[i-tenkan_period+1:i+1]) + np.min(low[i-tenkan_period+1:i+1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = np.full_like(high, np.nan)
    for i in range(kijun_period - 1, n):
        kijun_sen[i] = (np.max(high[i-kijun_period+1:i+1]) + np.min(low[i-kijun_period+1:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a = np.full_like(high, np.nan)
    for i in range(n):
        if i + kijun_period < n and not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
            senkou_span_a[i + kijun_period] = (tenkan_sen[i] + kijun_sen[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b = np.full_like(high, np.nan)
    for i in range(senkou_span_b_period - 1, n):
        if i + kijun_period < n:
            senkou_span_b[i + kijun_period] = (np.max(high[i-senkou_span_b_period+1:i+1]) + np.min(low[i-senkou_span_b_period+1:i+1])) / 2
    
    # Get 1d data for Cloud (Kumo) filter and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Ichimoku Cloud on 1d (same parameters)
    tenkan_1d = np.full_like(high_1d, np.nan)
    kijun_1d = np.full_like(high_1d, np.nan)
    senkou_a_1d = np.full_like(high_1d, np.nan)
    senkou_b_1d = np.full_like(high_1d, np.nan)
    
    for i in range(tenkan_period - 1, len(high_1d)):
        tenkan_1d[i] = (np.max(high_1d[i-tenkan_period+1:i+1]) + np.min(low_1d[i-tenkan_period+1:i+1])) / 2
    for i in range(kijun_period - 1, len(high_1d)):
        kijun_1d[i] = (np.max(high_1d[i-kijun_period+1:i+1]) + np.min(low_1d[i-kijun_period+1:i+1])) / 2
    for i in range(len(high_1d)):
        if i + kijun_period < len(high_1d) and not np.isnan(tenkan_1d[i]) and not np.isnan(kijun_1d[i]):
            senkou_a_1d[i + kijun_period] = (tenkan_1d[i] + kijun_1d[i]) / 2
    for i in range(senkou_span_b_period - 1, len(high_1d)):
        if i + kijun_period < len(high_1d):
            senkou_b_1d[i + kijun_period] = (np.max(high_1d[i-senkou_span_b_period+1:i+1]) + np.min(low_1d[i-senkou_span_b_period+1:i+1])) / 2
    
    # Calculate 1d Cloud boundaries (Senkou Span A and B)
    # Cloud top = max(Senkou A, Senkou B), Cloud bottom = min(Senkou A, Senkou B)
    cloud_top_1d = np.full_like(close_1d, np.nan)
    cloud_bottom_1d = np.full_like(close_1d, np.nan)
    for i in range(len(high_1d)):
        if not np.isnan(senkou_a_1d[i]) and not np.isnan(senkou_b_1d[i]):
            cloud_top_1d[i] = max(senkou_a_1d[i], senkou_b_1d[i])
            cloud_bottom_1d[i] = min(senkou_a_1d[i], senkou_b_1d[i])
    
    # Volume confirmation: 1d volume spike (2x 20-period MA)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean()
    vol_spike_1d = volume_1d > (vol_ma_1d.values * 2.0)
    
    # Align 1d indicators to 6s timeframe
    cloud_top_aligned = align_htf_to_ltf(prices, df_1d, cloud_top_1d)
    cloud_bottom_aligned = align_htf_to_ltf(prices, df_1d, cloud_bottom_1d)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(tenkan_period, kijun_period, senkou_span_b_period) + kijun_period  # Ensure Ichimoku is calculated
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(cloud_top_aligned[i]) or np.isnan(cloud_bottom_aligned[i]) or 
            np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Tenkan > Kijun (bullish cross) AND price above cloud AND volume spike
            if (tenkan_sen[i] > kijun_sen[i] and 
                close[i] > cloud_top_aligned[i] and 
                vol_spike_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Tenkan < Kijun (bearish cross) AND price below cloud AND volume spike
            elif (tenkan_sen[i] < kijun_sen[i] and 
                  close[i] < cloud_bottom_aligned[i] and 
                  vol_spike_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Tenkan < Kijun (bearish cross) OR price drops below cloud
            if (tenkan_sen[i] < kijun_sen[i] or 
                close[i] < cloud_top_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Tenkan > Kijun (bullish cross) OR price rises above cloud
            if (tenkan_sen[i] > kijun_sen[i] or 
                close[i] > cloud_bottom_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals