#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_TK_Cross_1dTrendFilter_Regime
Hypothesis: Ichimoku TK cross with 1d trend filter and chop regime on 6h timeframe. 
Long when TK crosses above in uptrend (price > cloud) and chop < 61.8 (trending), 
short when TK crosses below in downtrend (price < cloud) and chop < 61.8.
Exit on opposite TK cross or when chop > 61.8 (range) to avoid whipsaws.
Uses discrete sizing 0.25 to target 12-30 trades/year (50-120 total over 4 years).
Designed for both bull and bear markets via 1d trend filter and regime adaptation.
"""

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
    
    # Load 1d data ONCE before loop for HTF trend filter and cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h timeframe (wait for completed 1d bar)
    tenkan_sen_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_6h = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_6h = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_6h = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Calculate 6h chop regime (EWMA of True Range / ATR)
    # True Range
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # ATR(14) using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr = np.zeros_like(close)
    atr[0] = np.nan
    for i in range(1, len(tr)):
        if np.isnan(tr[i]) or np.isnan(atr[i-1]):
            atr[i] = np.nan
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Chop = (sum(atr(14)) / (max(high) - min(low)) over 14 periods) * 100
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = (atr_sum / (hh - ll + 1e-10)) * 100
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 52 for Ichimoku, 14 for chop, 26 for Senkou shift
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen_6h[i]) or
            np.isnan(kijun_sen_6h[i]) or
            np.isnan(senkou_span_a_6h[i]) or
            np.isnan(senkou_span_b_6h[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        tenkan = tenkan_sen_6h[i]
        kijun = kijun_sen_6h[i]
        span_a = senkou_span_a_6h[i]
        span_b = senkou_span_b_6h[i]
        chop_val = chop[i]
        
        # Determine cloud top and bottom
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        # Determine 1d trend: price vs Senkou Span B (longer term)
        trend_up = close_val > span_b
        trend_down = close_val < span_b
        
        if position == 0:
            # Flat - look for TK cross with filters
            # Bullish TK cross: Tenkan crosses above Kijun
            bullish_cross = tenkan > kijun and tenkan_sen_6h[i-1] <= kijun_sen_6h[i-1]
            # Bearish TK cross: Tenkan crosses below Kijun
            bearish_cross = tenkan < kijun and tenkan_sen_6h[i-1] >= kijun_sen_6h[i-1]
            
            # Only trade in trending regime (chop < 61.8) and with 1d trend filter
            if bullish_cross and trend_up and chop_val < 61.8:
                signals[i] = fixed_size
                position = 1
            elif bearish_cross and trend_down and chop_val < 61.8:
                signals[i] = -fixed_size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on bearish TK cross or when chop indicates range
            bearish_cross = tenkan < kijun and tenkan_sen_6h[i-1] >= kijun_sen_6h[i-1]
            if bearish_cross or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = fixed_size
        elif position == -1:
            # Short - exit on bullish TK cross or when chop indicates range
            bullish_cross = tenkan > kijun and tenkan_sen_6h[i-1] <= kijun_sen_6h[i-1]
            if bullish_cross or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -fixed_size
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_1dTrendFilter_Regime"
timeframe = "6h"
leverage = 1.0