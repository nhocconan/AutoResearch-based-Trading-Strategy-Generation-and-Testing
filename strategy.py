#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Ichimoku Cloud (Senkou Span A/B) for trend filter,
# Tenkan-Kijun cross for entry timing, and volume confirmation.
# Ichimoku works in both bull/bear markets: price above cloud = bullish bias, below = bearish.
# TK cross provides timely entries with reduced whipsaw vs MA cross.
# Volume filter ensures breakout strength. Target 12-30 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # need 26*2 for Ichimoku
        return np.zeros(n)
    
    # Calculate Ichimoku components on daily data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(df_1d['high'].values).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(df_1d['low'].values).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(df_1d['high'].values).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(df_1d['low'].values).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 plotted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    period52_high = pd.Series(df_1d['high'].values).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(df_1d['low'].values).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind (not used for signals)
    
    # Align HTF Ichimoku components to LTF (6h)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Calculate daily volume average for volume filter
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A/B)
        upper_cloud = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        lower_cloud = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Volume filter: current 6h volume > 1.5x daily average volume (scaled to 6h)
        # Approximate: 6h volume > (daily avg vol / 4) * 1.5 since 4x 6h bars per day
        vol_filter = volume[i] > (vol_ma_aligned[i] / 4.0) * 1.5
        
        # Long conditions:
        # 1. Price above cloud (bullish bias)
        # 2. Tenkan crosses above Kijun (bullish momentum)
        # 3. Volume confirmation
        if (close[i] > upper_cloud and 
            tenkan_aligned[i] > kijun_aligned[i] and
            tenkan_aligned[i-1] <= kijun_aligned[i-1] and  # cross just happened
            vol_filter):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below cloud (bearish bias)
        # 2. Tenkan crosses below Kijun (bearish momentum)
        # 3. Volume confirmation
        elif (close[i] < lower_cloud and 
              tenkan_aligned[i] < kijun_aligned[i] and
              tenkan_aligned[i-1] >= kijun_aligned[i-1] and  # cross just happened
              vol_filter):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Ichimoku_TK_Cross_Volume_Filter"
timeframe = "6h"
leverage = 1.0