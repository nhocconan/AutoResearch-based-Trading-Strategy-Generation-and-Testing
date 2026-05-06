#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Ichimoku cloud with TK cross and volume confirmation
# Long when Tenkan-sen crosses above Kijun-sen AND price is above cloud AND volume > 1.5 * avg_volume(20)
# Short when Tenkan-sen crosses below Kijun-sen AND price is below cloud AND volume > 1.5 * avg_volume(20)
# Exit when TK cross reverses or price touches cloud boundary
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Ichimoku provides dynamic support/resistance with trend, momentum and volatility in one system
# TK cross captures momentum shifts, cloud acts as dynamic support/resistance filter
# Volume confirmation ensures breakouts have participation, reducing false signals
# Works in bull (trend-following TK crosses above cloud) and bear (trend-following TK crosses below cloud)

name = "6h_1dIchimokuTKCross_CloudFilter_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need sufficient data for Ichimoku (26*2)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_series_1d = pd.Series(high_1d)
    low_series_1d = pd.Series(low_1d)
    tenkan_sen = (high_series_1d.rolling(window=9, min_periods=9).max() + 
                  low_series_1d.rolling(window=9, min_periods=9).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (high_series_1d.rolling(window=26, min_periods=26).max() + 
                 low_series_1d.rolling(window=26, min_periods=26).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b = ((high_series_1d.rolling(window=52, min_periods=52).max() + 
                      low_series_1d.rolling(window=52, min_periods=52).min()) / 2).shift(26)
    
    # Align Ichimoku components to 6h timeframe (wait for completed 1d bar)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud boundaries (Senkou Span A and B form the cloud)
        upper_cloud = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_cloud = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Check for TK cross
        tk_cross_above = (tenkan_sen_aligned[i] > kijun_sen_aligned[i] and 
                          tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1])
        tk_cross_below = (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and 
                          tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1])
        
        if position == 0:
            # Long: TK cross above AND price above cloud AND volume confirmation
            if (tk_cross_above and close[i] > upper_cloud and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: TK cross below AND price below cloud AND volume confirmation
            elif (tk_cross_below and close[i] < lower_cloud and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TK cross reverses below OR price touches lower cloud
            if (tk_cross_below or close[i] <= lower_cloud):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TK cross reverses above OR price touches upper cloud
            if (tk_cross_above or close[i] >= upper_cloud):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals