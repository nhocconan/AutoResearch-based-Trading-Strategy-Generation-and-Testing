#!/usr/bin/env python3
# 6h_ichimoku_1d_trend_v1
# Hypothesis: 6h strategy using Ichimoku cloud from 1d timeframe for trend direction and TK cross for entry timing.
# Enters long when price is above 1d Ichimoku cloud, Tenkan-sen crosses above Kijun-sen, and volume > 1.3x 20-period average.
# Enters short when price is below 1d Ichimoku cloud, Tenkan-sen crosses below Kijun-sen, and volume > 1.3x average.
# Uses discrete position sizing (±0.25) to minimize fee churn.
# Ichimoku provides strong trend filtering that works in both bull and bear markets via cloud position.
# TK cross gives timely entries within the trend. Volume confirmation reduces false signals.
# Target: 50-150 total trades over 4 years (12-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data ONCE before loop for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 for Ichimoku (26*2)
        return np.zeros(n)
    
    # Calculate Ichimoku components for daily
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
    
    # Align Ichimoku components to 6h timeframe (completed daily candle only)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Volume spike detection (20-period volume average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 1.3)  # Volume at least 1.3x average
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine Ichimoku cloud boundaries (Senkou Span A/B)
        upper_cloud = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_cloud = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        if position == 1:  # Long position
            # Exit: price falls below Ichimoku cloud OR Tenkan-sen crosses below Kijun-sen
            if (close[i] < lower_cloud) or (tenkan_sen_aligned[i] < kijun_sen_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above Ichimoku cloud OR Tenkan-sen crosses above Kijun-sen
            if (close[i] > upper_cloud) or (tenkan_sen_aligned[i] > kijun_sen_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price above cloud, Tenkan-sen crosses above Kijun-sen, with volume spike
            tenkan_prev = tenkan_sen_aligned[i-1] if i > 0 else tenkan_sen_aligned[i]
            kijun_prev = kijun_sen_aligned[i-1] if i > 0 else kijun_sen_aligned[i]
            tk_cross_up = (tenkan_prev <= kijun_prev) and (tenkan_sen_aligned[i] > kijun_sen_aligned[i])
            
            if (close[i] > upper_cloud) and tk_cross_up and vol_spike[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price below cloud, Tenkan-sen crosses below Kijun-sen, with volume spike
            elif (close[i] < lower_cloud) and (tenkan_prev >= kijun_prev) and (tenkan_sen_aligned[i] < kijun_sen_aligned[i]) and vol_spike[i]:
                position = -1
                signals[i] = -0.25
    
    return signals