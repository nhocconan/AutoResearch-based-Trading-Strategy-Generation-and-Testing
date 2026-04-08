#!/usr/bin/env python3
"""
6-hour Ichimoku Cloud with 1-day filter
Hypothesis: Ichimoku TK cross combined with 1-day price above/below cloud and volume confirmation provides
high-probability trend entries that work in both bull and bear markets by filtering weak signals.
Designed for ~15-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_1d_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for cloud calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Calculate Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                  pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    
    # Calculate Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                 pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Calculate Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = (pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                     pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2
    
    # Shift Senkou spans forward by 26 periods
    senkou_span_a_shifted = senkou_span_a.shift(kijun_period)
    senkou_span_b_shifted = senkou_span_b.shift(kijun_period)
    
    # Calculate 1-day Ichimoku cloud for trend filter
    tenkan_sen_1d = (pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                     pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    kijun_sen_1d = (pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                    pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    senkou_span_a_1d = ((tenkan_sen_1d + kijun_sen_1d) / 2)
    senkou_span_b_1d = (pd.Series(high_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                        pd.Series(low_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2
    
    # Determine cloud top and bottom for 1-day
    cloud_top_1d = np.maximum(senkou_span_a_1d, senkou_span_b_1d)
    cloud_bottom_1d = np.minimum(senkou_span_a_1d, senkou_span_b_1d)
    
    # Align 1-day cloud data to 6-hour timeframe
    cloud_top_1d_aligned = align_htf_to_ltf(prices, df_1d, cloud_top_1d)
    cloud_bottom_1d_aligned = align_htf_to_ltf(prices, df_1d, cloud_bottom_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_a_shifted[i]) or np.isnan(senkou_span_b_shifted[i]) or
            np.isnan(cloud_top_1d_aligned[i]) or np.isnan(cloud_bottom_1d_aligned[i]) or
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom (accounting for Senkou span shift)
        cloud_top = np.maximum(senkou_span_a_shifted[i], senkou_span_b_shifted[i])
        cloud_bottom = np.minimum(senkou_span_a_shifted[i], senkou_span_b_shifted[i])
        
        if position == 1:  # Long position
            # Exit: price drops below cloud OR TK cross turns bearish
            if (close[i] < cloud_bottom or tenkan_sen[i] < kijun_sen[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above cloud OR TK cross turns bullish
            if (close[i] > cloud_top or tenkan_sen[i] > kijun_sen[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # TK cross signals
            tk_cross_bullish = tenkan_sen[i] > kijun_sen[i]
            tk_cross_bearish = tenkan_sen[i] < kijun_sen[i]
            
            # Long: bullish TK cross + price above 1-day cloud + volume spike
            if (tk_cross_bullish and 
                close[i] > cloud_top_1d_aligned[i] and
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: bearish TK cross + price below 1-day cloud + volume spike
            elif (tk_cross_bearish and 
                  close[i] < cloud_bottom_1d_aligned[i] and
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals