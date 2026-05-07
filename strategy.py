#!/usr/bin/env python3
# 6h_IchimokuCloud_Trend_Follow
# Hypothesis: On 6h chart, use Ichimoku system for trend following:
# - Tenkan-sen (9) / Kijun-sen (26) cross for entry signals
# - Senkou Span A/B form cloud for trend filter (price above/below cloud)
# - Use 1d trend (Kijun-sen 26) to confirm higher timeframe bias
# - Works in both bull and bear by following established trends with cloud filter reducing false signals
# Target: 50-150 total trades over 4 years (12-37/year)
timeframe = "6h"
name = "6h_IchimokuCloud_Trend_Follow"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_high = pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max()
    tenkan_low = pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min()
    tenkan_sen = (tenkan_high + tenkan_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_high = pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max()
    kijun_low = pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min()
    kijun_sen = (kijun_high + kijun_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b_high = pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max()
    senkou_span_b_low = pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()
    senkou_span_b = (senkou_span_b_high + senkou_span_b_low) / 2
    
    # Chikou Span (Lagging Span): not used for signals (would look ahead)
    
    # Get 1d trend filter (Kijun-sen 26 on daily)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < kijun_period:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    kijun_high_1d = pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max()
    kijun_low_1d = pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min()
    kijun_sen_1d = (kijun_high_1d + kijun_low_1d) / 2
    kijun_sen_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen_1d)
    
    # Determine cloud boundaries (ahead by 26 periods)
    senkou_span_a_shifted = np.roll(senkou_span_a, kijun_period)
    senkou_span_b_shifted = np.roll(senkou_span_b, kijun_period)
    # Fill leading NaN values
    for i in range(kijun_period):
        senkou_span_a_shifted[i] = senkou_span_a[i] if not np.isnan(senkou_span_a[i]) else 0
        senkou_span_b_shifted[i] = senkou_span_b[i] if not np.isnan(senkou_span_b[i]) else 0
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_span_a_shifted, senkou_span_b_shifted)
    cloud_bottom = np.minimum(senkou_span_a_shifted, senkou_span_b_shifted)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(tenkan_period, kijun_period, senkou_span_b_period) + kijun_period, n):
        # Skip if any critical value is NaN
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(kijun_sen_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Price relative to cloud
        price_above_cloud = close[i] > cloud_top[i]
        price_below_cloud = close[i] < cloud_bottom[i]
        
        # TK Cross
        tk_cross_bull = tenkan_sen[i] > kijun_sen[i] and tenkan_sen[i-1] <= kijun_sen[i-1]
        tk_cross_bear = tenkan_sen[i] < kijun_sen[i] and tenkan_sen[i-1] >= kijun_sen[i-1]
        
        if position == 0:
            # Long: TK cross bull + price above cloud + 1d bullish bias
            if tk_cross_bull and price_above_cloud and close[i] > kijun_sen_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: TK cross bear + price below cloud + 1d bearish bias
            elif tk_cross_bear and price_below_cloud and close[i] < kijun_sen_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TK cross bear OR price falls below cloud
            if tk_cross_bear or close[i] < cloud_top[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TK cross bull OR price rises above cloud
            if tk_cross_bull or close[i] > cloud_bottom[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals