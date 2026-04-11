#!/usr/bin/env python3
# 6h_1d_ichimoku_cloud_trend_v1
# Strategy: 6h Ichimoku TK cross with 1d cloud filter
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Ichimoku provides trend direction and support/resistance. TK cross signals momentum shift. Cloud from higher timeframe (1d) filters for higher probability trades. Works in bull via bullish TK cross above cloud, in bear via bearish TK cross below cloud. Targets 15-30 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ichimoku_cloud_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d data ONCE before loop for cloud filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # 1d Ichimoku cloud (Senkou Span A and B)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    senkou_span_b = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                      pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2)
    
    # Align cloud components to 6h timeframe (with proper delay)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    # 6h Ichimoku components for TK cross
    # Tenkan-sen (6h): (9-period high + low)/2
    tenkan_sen_6h = (pd.Series(high).rolling(window=9, min_periods=9).max() + 
                     pd.Series(low).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (6h): (26-period high + low)/2
    kijun_sen_6h = (pd.Series(high).rolling(window=26, min_periods=26).max() + 
                    pd.Series(low).rolling(window=26, min_periods=26).min()) / 2
    
    tenkan_sen_6h_vals = tenkan_sen_6h.values
    kijun_sen_6h_vals = kijun_sen_6h.values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after 52 periods for full Ichimoku
        # Skip if any required data is invalid
        if (np.isnan(tenkan_sen_6h_vals[i]) or np.isnan(kijun_sen_6h_vals[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Cloud boundaries (Senkou Span A and B)
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # TK cross signals
        tk_bullish = tenkan_sen_6h_vals[i] > kijun_sen_6h_vals[i]
        tk_bearish = tenkan_sen_6h_vals[i] < kijun_sen_6h_vals[i]
        
        # Price relative to cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Entry logic: TK cross with cloud filter
        if tk_bullish and price_above_cloud and position != 1:
            position = 1
            signals[i] = 0.25
        elif tk_bearish and price_below_cloud and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite TK cross
        elif position == 1 and tk_bearish:
            position = 0
            signals[i] = 0.0
        elif position == -1 and tk_bullish:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals