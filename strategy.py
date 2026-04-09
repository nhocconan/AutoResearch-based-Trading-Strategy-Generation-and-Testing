#!/usr/bin/env python3
# 6h_ichimoku_cloud_trend_v1
# Hypothesis: 6h strategy using Ichimoku Cloud for trend identification with 1d HTF cloud filter and volume confirmation.
# Enters long when price is above both 6h and 1d clouds with bullish TK cross and volume > 1.5x 20-period average.
# Enters short when price is below both clouds with bearish TK cross and volume confirmation.
# Uses discrete position sizing (0.25) to limit fee drag.
# Target: 50-150 total trades over 4 years (12-37/year) to work in both bull and bear markets by following multi-timeframe trend structure.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
    """Calculate Ichimoku Cloud components"""
    if len(high) < kijun:
        nan_array = np.full_like(high, np.nan, dtype=float)
        return nan_array, nan_array, nan_array, nan_array
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period1_high = pd.Series(high).rolling(window=tenkan, min_periods=tenkan).max()
    period1_low = pd.Series(low).rolling(window=tenkan, min_periods=tenkan).min()
    tenkan_sen = (period1_high + period1_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period2_high = pd.Series(high).rolling(window=kijun, min_periods=kijun).max()
    period2_low = pd.Series(low).rolling(window=kijun, min_periods=kijun).min()
    kijun_sen = (period2_high + period2_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period3_high = pd.Series(high).rolling(window=senkou, min_periods=senkou).max()
    period3_low = pd.Series(low).rolling(window=senkou, min_periods=senkou).min()
    senkou_span_b = (period3_high + period3_low) / 2
    
    return tenkan_sen.values, kijun_sen.values, senkou_span_a.values, senkou_span_b.values

name = "6h_ichimoku_cloud_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Ichimoku on primary timeframe (6h)
    tenkan_6h, kijun_6h, senkou_a_6h, senkou_b_6h = calculate_ichimoku(high, low, close)
    
    # 1d HTF Ichimoku trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values
    )
    
    # Align 1d Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(tenkan_6h[i]) or
            np.isnan(kijun_6h[i]) or np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or
            np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Determine cloud boundaries (Senkou Span A/B)
        # Cloud top = max(Senkou A, Senkou B), Cloud bottom = min(Senkou A, Senkou B)
        cloud_top_6h = np.maximum(senkou_a_6h[i], senkou_b_6h[i])
        cloud_bottom_6h = np.minimum(senkou_a_6h[i], senkou_b_6h[i])
        cloud_top_1d = np.maximum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        cloud_bottom_1d = np.minimum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        # Bullish TK cross: Tenkan > Kijun
        tk_bullish_6h = tenkan_6h[i] > kijun_6h[i]
        tk_bullish_1d = tenkan_1d_aligned[i] > kijun_1d_aligned[i]
        tk_bearish_6h = tenkan_6h[i] < kijun_6h[i]
        tk_bearish_1d = tenkan_1d_aligned[i] < kijun_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below 6h cloud OR bearish TK cross on 6h
            if close[i] < cloud_top_6h or not tk_bullish_6h:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above 6h cloud OR bullish TK cross on 6h
            if close[i] > cloud_bottom_6h or tk_bullish_6h:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter only with volume confirmation and multi-timeframe alignment
            if volume_confirmed:
                # Bullish conditions: price above both clouds + bullish TK cross on both timeframes
                bullish_6h = close[i] > cloud_top_6h and tk_bullish_6h
                bullish_1d = close[i] > cloud_top_1d and tk_bullish_1d
                
                # Bearish conditions: price below both clouds + bearish TK cross on both timeframes
                bearish_6h = close[i] < cloud_bottom_6h and tk_bearish_6h
                bearish_1d = close[i] < cloud_bottom_1d and tk_bearish_1d
                
                # Long: bullish alignment across timeframes
                if bullish_6h and bullish_1d:
                    position = 1
                    signals[i] = 0.25
                # Short: bearish alignment across timeframes
                elif bearish_6h and bearish_1d:
                    position = -1
                    signals[i] = -0.25
    
    return signals