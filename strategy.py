#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1w Ichimoku cloud filter and 1d TK cross for trend direction
# Long when price above 1w Kumo (cloud) and TK cross bullish on 1d
# Short when price below 1w Kumo (cloud) and TK cross bearish on 1d
# Uses weekly Ichimoku for major trend filter and daily TK cross for entry timing
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data for Ichimoku cloud
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate Ichimoku components on weekly
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    tenkan_sen = (pd.Series(high_1w).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_1w).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + low)/2
    kijun_sen = (pd.Series(high_1w).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1w).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    senkou_span_b = ((pd.Series(high_1w).rolling(window=52, min_periods=52).max() + 
                      pd.Series(low_1w).rolling(window=52, min_periods=52).min()) / 2).shift(26)
    
    # Get 1d data for TK cross
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate TK cross on daily
    tenkan_sen_1d = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                     pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    kijun_sen_1d = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                    pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    tk_cross = tenkan_sen_1d - kijun_sen_1d  # Positive = bullish cross
    
    # Align indicators to 6h timeframe
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b.values)
    tk_cross_aligned = align_htf_to_ltf(prices, df_1d, tk_cross.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(senkou_span_a_aligned[i]) or 
            np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(tk_cross_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine if price is above or below Kumo (cloud)
        cloud_top = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # TK cross signals
        tk_bullish = tk_cross_aligned[i] > 0
        tk_bearish = tk_cross_aligned[i] < 0
        
        # Entry logic
        long_entry = price_above_cloud and tk_bullish
        short_entry = price_below_cloud and tk_bearish
        
        # Exit conditions: opposite signal
        exit_long = position == 1 and (price_below_cloud or tk_bearish)
        exit_short = position == -1 and (price_above_cloud or tk_bullish)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1w_1d_ichimoku_tk_cross_v1"
timeframe = "6h"
leverage = 1.0