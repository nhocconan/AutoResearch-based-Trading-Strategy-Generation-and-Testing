#!/usr/bin/env python3
# 6h_ichimoku_cloud_1d_v2
# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation.
# Uses Ichimoku TK cross for entry signals, price above/below 1d cloud for trend bias,
# and volume spike to confirm institutional participation. Designed for 12-37 trades/year.
# Works in bull/bear markets: Ichimoku provides dynamic support/resistance, 1d cloud filter avoids
# counter-trend trades during ranging, volume confirmation reduces false breakouts.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_1d_v2"
timeframe = "6h"
leverage = 1.0

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
    """Calculate Ichimoku Cloud components"""
    if len(high) < senkou:
        return (np.full_like(high, np.nan, dtype=float),
                np.full_like(high, np.nan, dtype=float),
                np.full_like(high, np.nan, dtype=float),
                np.full_like(high, np.nan, dtype=float))
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high).rolling(window=tenkan, min_periods=tenkan).max() +
                  pd.Series(low).rolling(window=tenkan, min_periods=tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high).rolling(window=kijun, min_periods=kijun).max() +
                 pd.Series(low).rolling(window=kijun, min_periods=kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = ((pd.Series(high).rolling(window=senkou, min_periods=senkou).max() +
                      pd.Series(low).rolling(window=senkou, min_periods=senkou).min()) / 2).shift(kijun)
    
    return (tenkan_sen.values, kijun_sen.values,
            senkou_span_a.values, senkou_span_b.values)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Ichimoku calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 104:  # Need enough for Ichimoku calculations
        return np.zeros(n)
    
    # Calculate 6h Ichimoku
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    tenkan_6h, kijun_6h, senkou_a_6h, senkou_b_6h = calculate_ichimoku(high_6h, low_6h, close_6h)
    
    # Align Ichimoku components to 6h timeframe (completed 6h candle only)
    tenkan_6h_aligned = align_htf_to_ltf(prices, df_6h, tenkan_6h)
    kijun_6h_aligned = align_htf_to_ltf(prices, df_6h, kijun_6h)
    senkou_a_6h_aligned = align_htf_to_ltf(prices, df_6h, senkou_a_6h)
    senkou_b_6h_aligned = align_htf_to_ltf(prices, df_6h, senkou_b_6h)
    
    # Get 1d HTF data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need enough for Ichimoku calculations
        return np.zeros(n)
    
    # Calculate 1d Ichimoku for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Align 1d Ichimoku components to 6h timeframe (completed daily candle only)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Volume spike detection (20-period volume average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_6h_aligned[i]) or np.isnan(kijun_6h_aligned[i]) or
            np.isnan(senkou_a_6h_aligned[i]) or np.isnan(senkou_b_6h_aligned[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Calculate cloud boundaries
        cloud_top_6h = np.maximum(senkou_a_6h_aligned[i], senkou_b_6h_aligned[i])
        cloud_bottom_6h = np.minimum(senkou_a_6h_aligned[i], senkou_b_6h_aligned[i])
        cloud_top_1d = np.maximum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        cloud_bottom_1d = np.minimum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        if position == 1:  # Long position
            # Exit: price closes below 6h cloud OR TK cross turns bearish
            if (close[i] < cloud_bottom_6h) or (tenkan_6h_aligned[i] < kijun_6h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 6h cloud OR TK cross turns bullish
            if (close[i] > cloud_top_6h) or (tenkan_6h_aligned[i] > kijun_6h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: bullish TK cross, price above 6h cloud, above 1d cloud, with volume spike
            if (tenkan_6h_aligned[i] > kijun_6h_aligned[i] and
                close[i] > cloud_top_6h and
                close[i] > cloud_top_1d and
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: bearish TK cross, price below 6h cloud, below 1d cloud, with volume spike
            elif (tenkan_6h_aligned[i] < kijun_6h_aligned[i] and
                  close[i] < cloud_bottom_6h and
                  close[i] < cloud_bottom_1d and
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals