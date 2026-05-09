#!/usr/bin/env python3
# 6h_Ichimoku_Cloud_Breakout_12hTrend
# Hypothesis: On 6h timeframe, enter long when price breaks above Kumo cloud and TK cross is bullish, with 12h trend filter. Enter short when price breaks below Kumo cloud and TK cross is bearish, with 12h trend filter.
# Uses 1d data for Ichimoku calculation (Tenkan-sen, Kijun-sen, Senkou Span A/B) to capture longer-term structure.
# Kumo cloud acts as dynamic support/resistance, TK cross provides momentum confirmation.
# Designed for 50-100 total trades over 4 years on 6h timeframe.

name = "6h_Ichimoku_Cloud_Breakout_12hTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 52:  # Need at least 52 periods for Ichimoku
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on 1d data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = np.full_like(high_1d, np.nan)
    period9_low = np.full_like(low_1d, np.nan)
    if len(high_1d) >= 9:
        for i in range(8, len(high_1d)):
            period9_high[i] = np.max(high_1d[i-8:i+1])
            period9_low[i] = np.min(low_1d[i-8:i+1])
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = np.full_like(high_1d, np.nan)
    period26_low = np.full_like(low_1d, np.nan)
    if len(high_1d) >= 26:
        for i in range(25, len(high_1d)):
            period26_high[i] = np.max(high_1d[i-25:i+1])
            period26_low[i] = np.min(low_1d[i-25:i+1])
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = np.full_like(high_1d, np.nan)
    period52_low = np.full_like(low_1d, np.nan)
    if len(high_1d) >= 52:
        for i in range(51, len(high_1d)):
            period52_high[i] = np.max(high_1d[i-51:i+1])
            period52_low[i] = np.min(low_1d[i-51:i+1])
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Get 12h data for trend filter (EMA 50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate 12h EMA(50) with proper initialization
    ema_50_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 50:
        ema_50_12h[49] = np.mean(close_12h[0:50])
        for i in range(50, len(close_12h)):
            ema_50_12h[i] = (close_12h[i] * 2 + ema_50_12h[i-1] * 48) / 50
    
    # Align 12h EMA to 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Need full Ichimoku calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Kumo cloud boundaries (Senkou Span A and B)
        upper_cloud = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_cloud = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # TK cross: Tenkan-sen > Kijun-sen is bullish
        tk_bullish = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        tk_bearish = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
        
        if position == 0:
            # Enter long: Price breaks above cloud AND TK cross bullish AND 12h trend bullish
            if (close[i] > upper_cloud and tk_bullish and 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below cloud AND TK cross bearish AND 12h trend bearish
            elif (close[i] < lower_cloud and tk_bearish and 
                  close[i] < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below cloud OR TK cross turns bearish OR trend turns bearish
            if (close[i] < lower_cloud or not tk_bullish or 
                close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above cloud OR TK cross turns bullish OR trend turns bullish
            if (close[i] > upper_cloud or not tk_bearish or 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals