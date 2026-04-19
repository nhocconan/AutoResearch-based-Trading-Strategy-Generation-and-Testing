#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud (Tenkan/Kijun) with 1d Cloud Filter
# - 6h Tenkan-sen (9) and Kijun-sen (26) cross for entry signals
# - 1d Kumo cloud (Senkou Span A/B) determines bullish/bearish regime
# - Only trade when 6h price is above/below 1d cloud in same direction
# - Volume confirmation: 6h volume > 1.5x 20-period average
# - Designed to work in both bull and bear markets by following higher timeframe cloud direction
# - Target: 12-25 trades/year to avoid excessive fee drag

name = "6h_Ichimoku_1dCloud_Filter_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for cloud filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Align 1d Ichimoku components to 6h
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Get 6h data for Tenkan/Kijun and volume
    df_6h = get_htf_data(prices, '6h')
    
    # 6h Ichimoku components
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high_6h = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
    period9_low_6h = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
    tenkan_sen_6h = (period9_high_6h + period9_low_6h) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high_6h = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
    period26_low_6h = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
    kijun_sen_6h = (period26_high_6h + period26_low_6h) / 2
    
    # 6h volume average (20-period)
    vol_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(vol_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    # Pre-compute session filter (00:00-23:00 UTC - trade all hours for 6h)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = np.ones(n, dtype=bool)  # Trade all hours for 6h timeframe
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i]) or
            np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(vol_ma_6h_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Cloud top and bottom
        cloud_top = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Volume filter: current 6h volume > 1.5x average
        volume_filter = vol_ma_6h_aligned[i] > 0 and volume[i] > 1.5 * vol_ma_6h_aligned[i]
        
        if position == 0:
            # Bullish conditions: price above cloud, Tenkan > Kijun, volume
            if (close[i] > cloud_top and 
                tenkan_sen_6h[i] > kijun_sen_6h[i] and 
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Bearish conditions: price below cloud, Tenkan < Kijun, volume
            elif (close[i] < cloud_bottom and 
                  tenkan_sen_6h[i] < kijun_sen_6h[i] and 
                  volume_filter):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price crosses below cloud or Tenkan < Kijun
            if close[i] < cloud_top or tenkan_sen_6h[i] < kijun_sen_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price crosses above cloud or Tenkan > Kijun
            if close[i] > cloud_bottom or tenkan_sen_6h[i] > kijun_sen_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals