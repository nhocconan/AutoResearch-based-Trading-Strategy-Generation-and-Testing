#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h volume-weighted price action with 1w Ichimoku cloud filter.
# Long when price crosses above 6h VWAP AND price > 1w Kumo cloud AND 6h volume > 1.5x 20-period average.
# Short when price crosses below 6h VWAP AND price < 1w Kumo cloud AND 6h volume > 1.5x 20-period average.
# Exit when price returns to 6h VWAP (mean reversion).
# Uses VWAP for intraday mean reversion with weekly Ichimoku trend filter to avoid counter-trend trades.
# Target: 80-160 total trades over 4 years (20-40/year) for balanced frequency and low fee drag.

name = "6h_VWAP_1wIchimoku_Cloud_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for Ichimoku cloud filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 6h VWAP (typical price * volume cumulative)
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = vwap_numerator / vwap_denominator
    
    # 6h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 1w Ichimoku cloud components
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_1w).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_1w).rolling(window=9, min_periods=9).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1w).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1w).rolling(window=26, min_periods=26).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b = ((pd.Series(high_1w).rolling(window=52, min_periods=52).max() + 
                      pd.Series(low_1w).rolling(window=52, min_periods=52).min()) / 2)
    
    # Align Ichimoku components to 6h timeframe (cloud requires Senkou Span A/B)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a.values, additional_delay_bars=26)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b.values, additional_delay_bars=26)
    
    # Kumo cloud boundaries (use aligned Senkou Spans)
    upper_cloud = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    lower_cloud = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Sufficient warmup for Ichimoku calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(vwap[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price crosses above VWAP, above Kumo cloud, volume spike
            long_cond = (close[i] > vwap[i]) and (close[i-1] <= vwap[i-1]) and \
                        (close[i] > upper_cloud[i]) and volume_filter[i]
            # Short conditions: price crosses below VWAP, below Kumo cloud, volume spike
            short_cond = (close[i] < vwap[i]) and (close[i-1] >= vwap[i-1]) and \
                         (close[i] < lower_cloud[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to VWAP (mean reversion)
            if close[i] <= vwap[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to VWAP (mean reversion)
            if close[i] >= vwap[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals