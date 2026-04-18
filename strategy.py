# 6h_1d_Ichimoku_TK_Cross_Cloud_Filter
# Hypothesis: 6h Ichimoku Tenkan-Kijun cross with 1d cloud filter for trend alignment.
# Long: TK cross bullish + price above 1d Kumo (cloud). Short: TK cross bearish + price below 1d Kumo.
# Uses cloud as dynamic support/resistance to avoid whipsaws in ranging markets.
# Works in both bull and bear by following cloud-filtered trend direction.
# Targets 50-150 total trades over 4 years via strict TK cross + cloud filter requirements.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for Ichimoku cloud (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Calculate Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_1d = np.full(len(high_1d), np.nan)
    for i in range(tenkan_period - 1, len(high_1d)):
        tenkan_1d[i] = (np.max(high_1d[i-tenkan_period+1:i+1]) + np.min(low_1d[i-tenkan_period+1:i+1])) / 2
    
    # Calculate Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_1d = np.full(len(high_1d), np.nan)
    for i in range(kijun_period - 1, len(high_1d)):
        kijun_1d[i] = (np.max(high_1d[i-kijun_period+1:i+1]) + np.min(low_1d[i-kijun_period+1:i+1])) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a_1d = np.full(len(high_1d), np.nan)
    for i in range(len(tenkan_1d)):
        if not np.isnan(tenkan_1d[i]) and not np.isnan(kijun_1d[i]):
            senkou_span_a_1d[i] = (tenkan_1d[i] + kijun_1d[i]) / 2
    
    # Calculate Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b_1d = np.full(len(high_1d), np.nan)
    for i in range(senkou_span_b_period - 1, len(high_1d)):
        senkou_span_b_1d[i] = (np.max(high_1d[i-senkou_span_b_period+1:i+1]) + np.min(low_1d[i-senkou_span_b_period+1:i+1])) / 2
    
    # Shift Senkou spans forward by 26 periods (cloud is plotted 26 periods ahead)
    # We'll handle alignment later with proper delay
    
    # Align Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_span_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d)
    senkou_span_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(tenkan_period, kijun_period, senkou_span_b_period) + 26  # Account for Senkou shift
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(senkou_span_a_1d_aligned[i]) or np.isnan(senkou_span_b_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Kumo (cloud) boundaries: Senkou Span A and B
        upper_cloud = np.maximum(senkou_span_a_1d_aligned[i], senkou_span_b_1d_aligned[i])
        lower_cloud = np.minimum(senkou_span_a_1d_aligned[i], senkou_span_b_1d_aligned[i])
        
        # TK cross signals
        tk_bullish = tenkan_1d_aligned[i] > kijun_1d_aligned[i]
        tk_bearish = tenkan_1d_aligned[i] < kijun_1d_aligned[i]
        
        if position == 0:
            # Long entry: TK cross bullish + price above cloud
            if tk_bullish and close[i] > upper_cloud:
                signals[i] = 0.25
                position = 1
            # Short entry: TK cross bearish + price below cloud
            elif tk_bearish and close[i] < lower_cloud:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: TK cross bearish OR price drops below cloud
            if tk_bearish or close[i] < lower_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TK cross bullish OR price rises above cloud
            if tk_bullish or close[i] > upper_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_Ichimoku_TK_Cross_Cloud_Filter"
timeframe = "6h"
leverage = 1.0