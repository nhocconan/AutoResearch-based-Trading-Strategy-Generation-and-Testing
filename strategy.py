#!/usr/bin/env python3
"""
6h_1d_1w_Ichimoku_Cloud_Breakout_v1
Hypothesis: Use Ichimoku cloud from daily timeframe as trend filter and support/resistance, with weekly higher-high/lower-low for bias confirmation. Enter on 6h breakout above/below cloud with volume confirmation. Works in bull/bear: In uptrend (price above cloud), buy breakouts; in downtrend (price below cloud), sell breakdowns. Weekly bias ensures alignment with longer-term structure.
Target: 12-25 trades/year per symbol (50-100 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 periods for Ichimoku
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components (standard periods: 9, 26, 52)
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
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind (not used for signals)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # The cloud: between Senkou Span A and B
    # Upper cloud = max(Senkou Span A, Senkou Span B)
    # Lower cloud = min(Senkou Span A, Senkou Span B)
    upper_cloud = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    lower_cloud = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    # Load 1w data for weekly bias: higher high/low structure
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 26:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly bias: 26-period higher high/low structure (simplified trend)
    # Bullish bias: current close > 26-period ago close AND current high > 26-period ago high
    # Bearish bias: current close < 26-period ago close AND current low < 26-period ago low
    close_1w_shifted = np.roll(close_1w, 26)
    high_1w_shifted = np.roll(high_1w, 26)
    low_1w_shifted = np.roll(low_1w, 26)
    close_1w_shifted[:26] = np.nan
    high_1w_shifted[:26] = np.nan
    low_1w_shifted[:26] = np.nan
    
    weekly_bullish = (close_1w > close_1w_shifted) & (high_1w > high_1w_shifted)
    weekly_bearish = (close_1w < close_1w_shifted) & (low_1w < low_1w_shifted)
    
    # Align weekly bias to 6h timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1d, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1d, weekly_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i]) or
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long conditions: price breaks above upper cloud AND weekly bullish bias AND volume
            if (price > upper_cloud[i] and 
                weekly_bullish_aligned[i] > 0.5 and  # Bullish bias
                volume_ok):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower cloud AND weekly bearish bias AND volume
            elif (price < lower_cloud[i] and 
                  weekly_bearish_aligned[i] > 0.5 and  # Bearish bias
                  volume_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price re-enters cloud (below lower cloud) or Tenkan-Kijun cross down
            if price < lower_cloud[i] or tenkan_sen_aligned[i] < kijun_sen_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price re-enters cloud (above upper cloud) or Tenkan-Kijun cross up
            if price > upper_cloud[i] or tenkan_sen_aligned[i] > kijun_sen_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_1w_Ichimoku_Cloud_Breakout_v1"
timeframe = "6h"
leverage = 1.0