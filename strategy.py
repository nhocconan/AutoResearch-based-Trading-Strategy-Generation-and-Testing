#!/usr/bin/env python3
# Hypothesis: 6h Williams %R + 1d Ichimoku Cloud Filter
# Long when Williams %R crosses above -20 (oversold bounce) and price above 1d Kumo cloud
# Short when Williams %R crosses below -80 (overbought rejection) and price below 1d Kumo cloud
# Exit when Williams %R returns to neutral zone (-50) or price crosses Tenkan/Kijun
# Uses Williams %R for momentum exhaustion and Ichimoku cloud for trend direction
# Designed to capture mean reversion within the prevailing daily trend
# Target: 60-120 total trades over 4 years (15-30/year) with size 0.25

name = "6h_WilliamsR_IchimokuCloud"
timeframe = "6h"
leverage = 1.0

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
    
    # Calculate 1d Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 days for Ichimoku
        return np.zeros(n)
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (df_1d['high'].rolling(window=9, min_periods=9).max() + 
                  df_1d['low'].rolling(window=9, min_periods=9).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (df_1d['high'].rolling(window=26, min_periods=26).max() + 
                 df_1d['low'].rolling(window=26, min_periods=26).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b = ((df_1d['high'].rolling(window=52, min_periods=52).max() + 
                      df_1d['low'].rolling(window=52, min_periods=52).min()) / 2).shift(26)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    # Calculate Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    williams_r = williams_r.values  # Convert to numpy array
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(williams_r[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud boundaries and direction
        upper_cloud = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_cloud = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Price above cloud = bullish, below cloud = bearish
        price_above_cloud = close[i] > upper_cloud
        price_below_cloud = close[i] < lower_cloud
        
        if position == 0:
            # Enter long: Williams %R crosses above -20 (from oversold) AND price above cloud
            if (williams_r[i] > -20 and williams_r[i-1] <= -20 and price_above_cloud):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R crosses below -80 (from overbought) AND price below cloud
            elif (williams_r[i] < -80 and williams_r[i-1] >= -80 and price_below_cloud):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns to neutral (-50) OR price crosses below Tenkan
            if williams_r[i] <= -50 or close[i] < tenkan_sen_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns to neutral (-50) OR price crosses above Tenkan
            if williams_r[i] >= -50 or close[i] > tenkan_sen_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals