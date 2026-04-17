#!/usr/bin/env python3
"""
6h_1d12h_Ichimoku_Cloud_Trend
Hypothesis: On 6h timeframe, trade Ichimoku cloud breakouts with daily/12h filters to capture strong trends. Uses 12h Kumo (cloud) as trend filter and 1d volume confirmation to avoid false signals. Designed for low trade frequency (15-35/year) to minimize fee drift while working in both bull and bear markets by requiring cloud alignment and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
    """
    Calculate Ichimoku components:
    Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    Kijun-sen (Base Line): (26-period high + 26-period low)/2
    Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    """
    # Tenkan-sen
    high_tenkan = pd.Series(high).rolling(window=tenkan, min_periods=tenkan).max()
    low_tenkan = pd.Series(low).rolling(window=tenkan, min_periods=tenkan).min()
    tenkan_sen = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen
    high_kijun = pd.Series(high).rolling(window=kijun, min_periods=kijun).max()
    low_kijun = pd.Series(low).rolling(window=kijun, min_periods=kijun).min()
    kijun_sen = (high_kijun + low_kijun) / 2
    
    # Senkou Span A
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    
    # Senkou Span B
    high_senkou = pd.Series(high).rolling(window=senkou, min_periods=senkou).max()
    low_senkou = pd.Series(low).rolling(window=senkou, min_periods=senkou).min()
    senkou_b = ((high_senkou + low_senkou) / 2).shift(kijun)
    
    # Current Chikou Span (lagging line) - not used for entry but for completeness
    # chikou_span = close.shift(-kijun)
    
    return tenkan_sen.values, kijun_sen.values, senkou_a.values, senkou_b.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 12h Data (HTF for Ichimoku cloud - trend filter) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Ichimoku on 12h
    tenkan_12h, kijun_12h, senkou_a_12h, senkou_b_12h = calculate_ichimoku(high_12h, low_12h, close_12h)
    
    # Align Ichimoku components
    tenkan_12h_aligned = align_htf_to_ltf(prices, df_12h, tenkan_12h)
    kijun_12h_aligned = align_htf_to_ltf(prices, df_12h, kijun_12h)
    senkou_a_12h_aligned = align_htf_to_ltf(prices, df_12h, senkou_a_12h)
    senkou_b_12h_aligned = align_htf_to_ltf(prices, df_12h, senkou_b_12h)
    
    # Cloud top and bottom
    cloud_top_12h = np.maximum(senkou_a_12h_aligned, senkou_b_12h_aligned)
    cloud_bottom_12h = np.minimum(senkou_a_12h_aligned, senkou_b_12h_aligned)
    
    # === 1d Data (for volume filter) ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Daily average volume (20-period)
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    
    # Warmup period - need enough data for Ichimoku calculations
    warmup = 52 + 26  # senkou period + kijun period for full Ichimoku
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_12h_aligned[i]) or 
            np.isnan(kijun_12h_aligned[i]) or
            np.isnan(cloud_top_12h[i]) or
            np.isnan(cloud_bottom_12h[i]) or
            np.isnan(vol_avg_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current daily bar's volume for confirmation
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        
        # Volume filter: current volume > 1.5x daily average volume
        vol_filter = vol_1d_current > 1.5 * vol_avg_1d_aligned[i]
        
        # Ichimoku trend filters
        # Bullish: price above cloud AND Tenkan > Kijun
        bullish_trend = (close[i] > cloud_top_12h[i]) and (tenkan_12h_aligned[i] > kijun_12h_aligned[i])
        
        # Bearish: price below cloud AND Tenkan < Kijun
        bearish_trend = (close[i] < cloud_bottom_12h[i]) and (tenkan_12h_aligned[i] < kijun_12h_aligned[i])
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: bullish trend + volume filter
            if bullish_trend and vol_filter:
                signals[i] = 0.25
                position = 1
                continue
            # Short: bearish trend + volume filter
            elif bearish_trend and vol_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit when price closes below cloud OR Tenkan < Kijun (trend change)
            if (close[i] < cloud_top_12h[i]) or (tenkan_12h_aligned[i] < kijun_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price closes above cloud OR Tenkan > Kijun (trend change)
            if (close[i] > cloud_bottom_12h[i]) or (tenkan_12h_aligned[i] > kijun_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d12h_Ichimoku_Cloud_Trend"
timeframe = "6h"
leverage = 1.0