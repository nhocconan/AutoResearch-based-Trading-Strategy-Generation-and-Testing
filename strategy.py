#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with TK cross and 1d EMA200 trend filter
# Long when Tenkan > Kijun (bullish TK cross) AND price > Cloud AND 1d EMA200 up
# Short when Tenkan < Kijun (bearish TK cross) AND price < Cloud AND 1d EMA200 down
# Exit when TK cross reverses or price re-enters Cloud
# Uses discrete position sizing (0.25) to reduce fee drag.
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to avoid overtrading.
# Ichimoku Cloud provides dynamic support/resistance with trend direction.
# TK cross gives timely entry signals while Cloud filter ensures trend alignment.
# 1d EMA200 ensures we only trade with the higher timeframe trend.

name = "6h_Ichimoku_Cloud_TK_Cross_1dEMA200_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    displacement = 26
    
    # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past 9 periods
    highest_tenkan = pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max()
    lowest_tenkan = pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min()
    tenkan = (highest_tenkan + lowest_tenkan) / 2
    
    # Calculate Kijun-sen (Base Line): (highest high + lowest low)/2 for past 26 periods
    highest_kijun = pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max()
    lowest_kijun = pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min()
    kijun = (highest_kijun + lowest_kijun) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_span_a = (tenkan + kijun) / 2
    
    # Calculate Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past 52 periods plotted 26 periods ahead
    highest_senkou_b = pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max()
    lowest_senkou_b = pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()
    senkou_span_b = (highest_senkou_b + lowest_senkou_b) / 2
    
    # The Cloud (Kumo) is between Senkou Span A and Senkou Span B
    # For simplicity, we'll use the current Cloud values (already displaced)
    # In real Ichimoku, Senkou spans are plotted ahead, but for our logic we check if price is above/below the Cloud
    # We'll calculate the Cloud boundaries without displacement for current price comparison
    highest_senkou_b_current = pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max()
    lowest_senkou_b_current = pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()
    senkou_span_b_current = (highest_senkou_b_current + lowest_senkou_b_current) / 2
    senkou_span_a_current = (tenkan + kijun) / 2
    
    # Upper Cloud boundary = max(Senkou A, Senkou B)
    # Lower Cloud boundary = min(Senkou A, Senkou B)
    upper_cloud = np.maximum(senkou_span_a_current.values, senkou_span_b_current.values)
    lower_cloud = np.minimum(senkou_span_a_current.values, senkou_span_b_current.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(tenkan_period, kijun_period, senkou_span_b_period, 200)  # warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(tenkan.iloc[i]) or 
            np.isnan(kijun.iloc[i]) or np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i])):
            signals[i] = 0.0
            continue
        
        curr_ema200_1d = ema_200_1d_aligned[i]
        curr_tenkan = tenkan.iloc[i]
        curr_kijun = kijun.iloc[i]
        curr_close = close[i]
        curr_upper_cloud = upper_cloud[i]
        curr_lower_cloud = lower_cloud[i]
        
        # Determine TK cross
        tk_bullish = curr_tenkan > curr_kijun
        tk_bearish = curr_tenkan < curr_kijun
        
        # Determine price relative to Cloud
        price_above_cloud = curr_close > curr_upper_cloud
        price_below_cloud = curr_close < curr_lower_cloud
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: TK cross turns bearish OR price re-enters Cloud
            if not tk_bullish or not price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TK cross turns bullish OR price re-enters Cloud
            if not tk_bearish or not price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when bullish TK cross AND price above Cloud AND 1d EMA200 up
            if tk_bullish and price_above_cloud and curr_close > curr_ema200_1d:
                signals[i] = 0.25
                position = 1
            # Short when bearish TK cross AND price below Cloud AND 1d EMA200 down
            elif tk_bearish and price_below_cloud and curr_close < curr_ema200_1d:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals