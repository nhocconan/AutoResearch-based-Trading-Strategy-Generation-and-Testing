#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = np.full_like(high_1d, np.nan)
    period9_low = np.full_like(low_1d, np.nan)
    for i in range(8, len(high_1d)):
        period9_high[i] = np.max(high_1d[i-8:i+1])
        period9_low[i] = np.min(low_1d[i-8:i+1])
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = np.full_like(high_1d, np.nan)
    period26_low = np.full_like(low_1d, np.nan)
    for i in range(25, len(high_1d)):
        period26_high[i] = np.max(high_1d[i-25:i+1])
        period26_low[i] = np.min(low_1d[i-25:i+1])
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    # Shift 26 periods ahead for plotting (but we'll use current values for cloud)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = np.full_like(high_1d, np.nan)
    period52_low = np.full_like(low_1d, np.nan)
    for i in range(51, len(high_1d)):
        period52_high[i] = np.max(high_1d[i-51:i+1])
        period52_low[i] = np.min(low_1d[i-51:i+1])
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou_span = np.full_like(close_1d, np.nan)
    for i in range(26, len(close_1d)):
        chikou_span[i] = close_1d[i-26]
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_6h = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_6h = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_6h = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    chikou_span_6h = align_htf_to_ltf(prices, df_1d, chikou_span)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after all indicators are valid
    start_idx = max(52, 26) + 1  # Need 52 periods for Senkou B
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i]) or 
            np.isnan(senkou_span_a_6h[i]) or np.isnan(senkou_span_b_6h[i]) or 
            np.isnan(chikou_span_6h[i])):
            signals[i] = 0.0
            continue
        
        # Ichimoku signals:
        # Cloud top and bottom
        cloud_top = max(senkou_span_a_6h[i], senkou_span_b_6h[i])
        cloud_bottom = min(senkou_span_a_6h[i], senkou_span_b_6h[i])
        
        # Price above/below cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # TK Cross (Tenkan-sen/Kijun-sen crossover)
        tk_cross_bull = tenkan_sen_6h[i] > kijun_sen_6h[i]
        tk_cross_bear = tenkan_sen_6h[i] < kijun_sen_6h[i]
        
        # Chikou Span confirmation (price vs 26 periods ago)
        chikou_confirm_bull = close[i] > chikou_span_6h[i]
        chikou_confirm_bear = close[i] < chikou_span_6h[i]
        
        if position == 0:
            # Long: Price above cloud + TK cross bullish + Chikou confirmation
            if price_above_cloud and tk_cross_bull and chikou_confirm_bull:
                signals[i] = 0.25
                position = 1
            # Short: Price below cloud + TK cross bearish + Chikou confirmation
            elif price_below_cloud and tk_cross_bear and chikou_confirm_bear:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price falls below cloud OR TK cross bearish
            if price_below_cloud or not tk_cross_bull:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises above cloud OR TK cross bullish
            if price_above_cloud or not tk_cross_bear:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_Chikou"
timeframe = "6h"
leverage = 1.0