#!/usr/bin/env python3
# 4h_1d_Ichimoku_Cloud_Breakout_Volume_Filter
# Hypothesis: Use 1d Ichimoku Cloud (Senkou Span A/B) with 4h price breakout and volume confirmation.
# In bull markets: price above cloud + bullish TK cross = long.
# In bear markets: price below cloud + bearish TK cross = short.
# Uses 1d cloud for major trend, 4h for entry timing. Targets 20-40 trades/year.
# Ichimoku provides dynamic support/resistance and trend direction, reducing false breaks.

name = "4h_1d_Ichimoku_Cloud_Breakout_Volume_Filter"
timeframe = "4h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate 1d Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods back (not used for signals)
    
    # Align Ichimoku components to 4h timeframe
    tenkan_sen_4h = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_4h = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_4h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_4h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Calculate cloud boundaries (Senkou Span A/B)
    upper_cloud = np.maximum(senkou_a_4h, senkou_b_4h)
    lower_cloud = np.minimum(senkou_a_4h, senkou_b_4h)
    
    # Volume average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_4h[i]) or np.isnan(kijun_sen_4h[i]) or 
            np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price above cloud + bullish TK cross + volume spike
            if (close[i] > upper_cloud[i] and 
                tenkan_sen_4h[i] > kijun_sen_4h[i] and 
                volume[i] > 2.0 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price below cloud + bearish TK cross + volume spike
            elif (close[i] < lower_cloud[i] and 
                  tenkan_sen_4h[i] < kijun_sen_4h[i] and 
                  volume[i] > 2.0 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price drops below cloud or bearish TK cross
            if close[i] < lower_cloud[i] or tenkan_sen_4h[i] < kijun_sen_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above cloud or bullish TK cross
            if close[i] > upper_cloud[i] or tenkan_sen_4h[i] > kijun_sen_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals