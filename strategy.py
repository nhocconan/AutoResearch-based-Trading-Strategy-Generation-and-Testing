#!/usr/bin/env python3
# 6h_Ichimoku_Cloud_Breakout_Daily_Trend
# Hypothesis: 6h Ichimoku cloud breakout with daily trend filter (price above/below 200 EMA).
# In bull markets, price tends to stay above 200 EMA and break above Ichimoku cloud for longs.
# In bear markets, price tends to stay below 200 EMA and break below Ichimoku cloud for shorts.
# The daily EMA200 filter ensures we only trade in the direction of the higher timeframe trend,
# reducing whipsaws during ranging periods. Ichimoku provides dynamic support/resistance.
# Target: 50-150 total trades over 4 years (12-37/year) with strict entry conditions.

name = "6h_Ichimoku_Cloud_Breakout_Daily_Trend"
timeframe = "6h"
leverage = 1.0

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
    
    # Calculate Ichimoku components on 6h data
    # Conversion Line (Tenkan-sen): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Base Line (Kijun-sen): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Leading Span A (Senkou Span A): (Conversion Line + Base Line) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Leading Span B (Senkou Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Get daily data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA200 on daily close
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), senkou_span_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 200)  # Ensure enough data for Ichomoku (52) and EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A and B)
        upper_cloud = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_cloud = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Check if price is above or below cloud
        price_above_cloud = close[i] > upper_cloud
        price_below_cloud = close[i] < lower_cloud
        
        # Daily trend filter: price relative to EMA200
        price_above_ema200 = close[i] > ema200_1d_aligned[i]
        price_below_ema200 = close[i] < ema200_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above cloud AND price above daily EMA200 (bullish alignment)
            if price_above_cloud and price_above_ema200:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below cloud AND price below daily EMA200 (bearish alignment)
            elif price_below_cloud and price_below_ema200:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below cloud OR price drops below daily EMA200
            if (price_below_cloud) or (not price_above_ema200):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above cloud OR price rises above daily EMA200
            if (price_above_cloud) or (not price_below_ema200):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals