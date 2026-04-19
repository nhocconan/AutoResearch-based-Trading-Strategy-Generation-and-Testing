#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Ichimoku_CloudBreakout_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Ichimoku Cloud components on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min()
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min()
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min()
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 4h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    # Volume spike filter: 1d volume > 2x 20-period average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 70  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 4h volume > 2x 1d average volume (scaled)
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 2.0 * (vol_ma_1d_aligned[i] / 6.0)
        
        if position == 0:
            # Cloud is bullish when Senkou Span A > Senkou Span B
            cloud_bullish = senkou_span_a_aligned[i] > senkou_span_b_aligned[i]
            # Cloud is bearish when Senkou Span A < Senkou Span B
            cloud_bearish = senkou_span_a_aligned[i] < senkou_span_b_aligned[i]
            
            # Long when price breaks above cloud and cloud is bullish
            if (close[i] > senkou_span_a_aligned[i] and 
                close[i] > senkou_span_b_aligned[i] and
                cloud_bullish and 
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short when price breaks below cloud and cloud is bearish
            elif (close[i] < senkou_span_a_aligned[i] and 
                  close[i] < senkou_span_b_aligned[i] and
                  cloud_bearish and 
                  volume_filter):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Exit long when price falls below Kijun-sen or cloud turns bearish
            if (close[i] < kijun_sen_aligned[i] or 
                senkou_span_a_aligned[i] < senkou_span_b_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Exit short when price rises above Tenkan-sen or cloud turns bullish
            if (close[i] > tenkan_sen_aligned[i] or 
                senkou_span_a_aligned[i] > senkou_span_b_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals