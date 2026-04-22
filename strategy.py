#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud breakout with weekly trend filter and volume confirmation.
Long when price breaks above Kumo (cloud) with weekly trend bullish (price > weekly Kijun-Sen) and volume spike.
Short when price breaks below Kumo with weekly trend bearish (price < weekly Kijun-Sen) and volume spike.
Exit when price re-enters the cloud.
Ichimoku provides dynamic support/resistance and trend direction, effective in both trending and ranging markets.
Weekly filter ensures alignment with higher timeframe trend to avoid counter-trend whipsaws.
Designed for low trade frequency (12-30/year) to minimize fee drag.
"""
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
    
    # Load weekly data for trend filter - ONCE before loop
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 26:
        return np.zeros(n)
    
    # Calculate weekly Ichimoku components for trend filter
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    tenkan_sen_w = (pd.Series(high_w).rolling(window=9, min_periods=9).max() + 
                    pd.Series(low_w).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + low)/2
    kijun_sen_w = (pd.Series(high_w).rolling(window=26, min_periods=26).max() + 
                   pd.Series(low_w).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a_w = ((tenkan_sen_w + kijun_sen_w) / 2)
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    senkou_span_b_w = ((pd.Series(high_w).rolling(window=52, min_periods=52).max() + 
                        pd.Series(low_w).rolling(window=52, min_periods=52).min()) / 2)
    
    # Align weekly components to 6h timeframe
    kijun_sen_w_aligned = align_htf_to_ltf(prices, df_w, kijun_sen_w.values)
    senkou_span_a_w_aligned = align_htf_to_ltf(prices, df_w, senkou_span_a_w.values)
    senkou_span_b_w_aligned = align_htf_to_ltf(prices, df_w, senkou_span_b_w.values)
    
    # Calculate 6h Ichimoku components for entry signals
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    tenkan_sen = (pd.Series(high).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + low)/2
    kijun_sen = (pd.Series(high).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    senkou_span_b = ((pd.Series(high).rolling(window=52, min_periods=52).max() + 
                      pd.Series(low).rolling(window=52, min_periods=52).min()) / 2)
    
    # Calculate 6h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after Senkou B lookback
        # Skip if data not ready
        if (np.isnan(kijun_sen_w_aligned[i]) or np.isnan(senkou_span_a_w_aligned[i]) or 
            np.isnan(senkou_span_b_w_aligned[i]) or np.isnan(tenkan_sen[i]) or 
            np.isnan(kijun_sen[i]) or np.isnan(senkou_span_a[i]) or 
            np.isnan(senkou_span_b[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud boundaries (Senkou Span A and B)
        upper_cloud = np.maximum(senkou_span_a[i], senkou_span_b[i])
        lower_cloud = np.minimum(senkou_span_a[i], senkou_span_b[i])
        
        if position == 0:
            # Long: Price breaks above cloud with bullish weekly trend and volume spike
            if (close[i] > upper_cloud and 
                close[i] > kijun_sen_w_aligned[i] and  # Weekly trend bullish: price above weekly Kijun-sen
                volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below cloud with bearish weekly trend and volume spike
            elif (close[i] < lower_cloud and 
                  close[i] < kijun_sen_w_aligned[i] and  # Weekly trend bearish: price below weekly Kijun-sen
                  volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price re-enters the cloud
            exit_signal = False
            
            if position == 1:
                # Exit long: price falls back into cloud
                if close[i] <= upper_cloud:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price rises back into cloud
                if close[i] >= lower_cloud:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Ichimoku_Cloud_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0
#%%