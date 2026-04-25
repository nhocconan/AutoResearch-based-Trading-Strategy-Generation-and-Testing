#!/usr/bin/env python3
"""
6h_ADX_Ichimoku_CloudTrend_v1
Hypothesis: Combine ADX trend strength with Ichimoku cloud filter on 6h timeframe. 
Enter long when ADX > 25 (strong trend) + price above Ichimoku cloud + bullish TK cross.
Enter short when ADX > 25 + price below Ichimoku cloud + bearish TK cross.
Use 1d Ichimoku for higher timeframe cloud alignment to reduce false signals.
Exit on TK cross reversal or ADX < 20 (trend weakening).
Position size: 0.25 to balance risk and return.
Target: 50-150 total trades over 4 years = 12-37/year.
Ichimoku cloud acts as dynamic support/resistance, ADX filters ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for Ichimoku HTF
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need sufficient data for Ichimoku (26*2)
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_9 = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_1d = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_26 = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_1d = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a_1d = ((tenkan_1d + kijun_1d) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_52 = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b_1d = ((max_high_52 + min_low_52) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou_1d = close_1d  # Will be aligned properly with shift
    
    # Align Ichimoku components to 6h prices
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_span_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d)
    senkou_span_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d)
    chikou_1d_aligned = align_htf_to_ltf(prices, df_1d, chikou_1d)
    
    # Calculate ADX on 6h
    period_adx = 14
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        elif low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    # True Range
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=period_adx, min_periods=period_adx).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=period_adx, min_periods=period_adx).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=period_adx, min_periods=period_adx).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=period_adx, min_periods=period_adx).mean().values
    
    # TK Cross (Tenkan/Kijun cross)
    tk_cross_bullish = tenkan_1d_aligned > kijun_1d_aligned
    tk_cross_bearish = tenkan_1d_aligned < kijun_1d_aligned
    
    # Price relative to cloud
    cloud_top = np.maximum(senkou_span_a_1d_aligned, senkou_span_b_1d_aligned)
    cloud_bottom = np.minimum(senkou_span_a_1d_aligned, senkou_span_b_1d_aligned)
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for ADX (14*2) and Ichimoku (52)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx[i]) or 
            np.isnan(tenkan_1d_aligned[i]) or
            np.isnan(kijun_1d_aligned[i]) or
            np.isnan(cloud_top[i]) or
            np.isnan(cloud_bottom[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # ADX trend strength
        strong_trend = adx[i] > 25
        weakening_trend = adx[i] < 20
        
        if position == 0:
            # Long setup: strong trend + price above cloud + bullish TK cross
            long_setup = strong_trend and price_above_cloud[i] and tk_cross_bullish[i]
            
            # Short setup: strong trend + price below cloud + bearish TK cross
            short_setup = strong_trend and price_below_cloud[i] and tk_cross_bearish[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: TK cross turns bearish OR trend weakens
            if tk_cross_bearish[i] or weakening_trend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: TK cross turns bullish OR trend weakens
            if tk_cross_bullish[i] or weakening_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ADX_Ichimoku_CloudTrend_v1"
timeframe = "6h"
leverage = 1.0