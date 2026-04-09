#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Ichimoku cloud for trend direction and 1d Camarilla pivots for entry/exit
# Weekly Ichimoku cloud (from 1w data) provides major trend filter: price above cloud = bullish bias, below = bearish bias
# 1d Camarilla pivots provide precise intraday support/resistance levels for entries
# Volume confirmation (current 6h volume > 1.8x 24-period average) filters false breakouts
# Designed for 6h timeframe targeting 12-30 trades/year (48-120 over 4 years)
# Works in bull/bear: Ichimoku cloud adapts to long-term trend, Camarilla levels provide precise entries, volume confirms validity

name = "6h_1w_1d_ichimoku_camarilla_volume_v1"
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
    
    # Load weekly data ONCE before loop for Ichimoku
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w < 52):
        return np.zeros(n)
    
    # Load daily data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    # Calculate weekly Ichimoku components
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2.0
    
    # Chikou Span (Lagging Span): not used for trend filter
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen)
    span_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a)
    span_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Key levels for trading: R3, R4, S3, S4 (stronger levels)
    camarilla_r3 = close_1d + range_1d * 1.1 / 4.0
    camarilla_r4 = close_1d + range_1d * 1.1 / 2.0
    camarilla_s3 = close_1d - range_1d * 1.1 / 4.0
    camarilla_s4 = close_1d - range_1d * 1.1 / 2.0
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Pre-compute volume confirmation (24-period average for 6h)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(span_a_aligned[i]) or np.isnan(span_b_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Determine Ichimoku trend: price above/below cloud
        # Cloud top = max(span_a, span_b), cloud bottom = min(span_a, span_b)
        cloud_top = np.maximum(span_a_aligned[i], span_b_aligned[i])
        cloud_bottom = np.minimum(span_a_aligned[i], span_b_aligned[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Volume confirmation: current 6h volume > 1.8x average 6h volume
        volume_confirmed = volume[i] > 1.8 * vol_ma_24[i]
        
        if position == 0:  # Flat - look for new entries
            if volume_confirmed:
                # Long: price above cloud AND breaking above Camarilla R4
                if price_above_cloud and close[i] > r4_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price below cloud AND breaking below Camarilla S4
                elif price_below_cloud and close[i] < s4_aligned[i]:
                    position = -1
                    signals[i] = -0.25
        
        elif position == 1:  # Long position - exit on cloud cross or S3 touch
            # Exit if price falls below cloud (trend change) or touches S3 (support)
            if close[i] < cloud_bottom or close[i] <= s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position - exit on cloud cross or R3 touch
            # Exit if price rises above cloud (trend change) or touches R3 (resistance)
            if close[i] > cloud_top or close[i] >= r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals