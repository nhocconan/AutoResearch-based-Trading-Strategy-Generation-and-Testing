#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_12hTrend_Filter_v2
Hypothesis: Ichimoku cloud with TK cross signals filtered by 12h trend (price > Kumo twist) on 6h timeframe.
Long when price breaks above cloud, Tenkan > Kijun, and 12h close > 12h EMA50.
Short when price breaks below cloud, Tenkan < Kijun, and 12h close < 12h EMA50.
Uses cloud thickness as volatility filter to avoid whipsaws. Designed for 6h to capture medium-term trends in both bull and bear markets.
Target trades: 12-30/year (48-120 total over 4 years) to minimize fee drag.
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Ichimoku parameters (9, 26, 52)
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    displacement = 26
    
    # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past 9 periods
    highest_high_9 = pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max().values
    lowest_low_9 = pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min().values
    tenkan = (highest_high_9 + lowest_low_9) / 2
    
    # Calculate Kijun-sen (Base Line): (highest high + lowest low)/2 for past 26 periods
    highest_high_26 = pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max().values
    lowest_low_26 = pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min().values
    kijun = (highest_high_26 + lowest_low_26) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan + kijun) / 2)
    
    # Calculate Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past 52 periods shifted 26 periods ahead
    highest_high_52 = pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max().values
    lowest_low_52 = pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min().values
    senkou_span_b = ((highest_high_52 + lowest_low_52) / 2)
    
    # Current Kumo (cloud) boundaries: Senkou Span A/B from 26 periods ago
    senkou_span_a_lag = np.roll(senkou_span_a, displacement)
    senkou_span_b_lag = np.roll(senkou_span_b, displacement)
    # Fill first displacement values with NaN
    senkou_span_a_lag[:displacement] = np.nan
    senkou_span_b_lag[:displacement] = np.nan
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_span_a_lag, senkou_span_b_lag)
    cloud_bottom = np.minimum(senkou_span_a_lag, senkou_span_b_lag)
    cloud_thickness = cloud_top - cloud_bottom
    
    # Align 12h EMA50 trend filter
    # No additional delay needed for EMA as it's based on completed 12h bar
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of Ichimoku calculations (52) + displacement (26) + 12h EMA (50)
    start_idx = tenkan_period + kijun_period + senkou_span_b_period + displacement  # 9+26+52+26 = 113
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or 
            np.isnan(cloud_thickness[i]) or np.isnan(ema_50_12h_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        close_val = close[i]
        cloud_top_val = cloud_top[i]
        cloud_bottom_val = cloud_bottom[i]
        cloud_thick_val = cloud_thickness[i]
        ema_12h_val = ema_50_12h_aligned[i]
        
        if position == 0:
            # Long: price breaks above cloud, Tenkan > Kijun, 12h uptrend, cloud thick enough
            long_signal = (close_val > cloud_top_val) and (tenkan_val > kijun_val) and (ema_12h_val > close_val) and (cloud_thick_val > 0)
            # Short: price breaks below cloud, Tenkan < Kijun, 12h downtrend, cloud thick enough
            short_signal = (close_val < cloud_bottom_val) and (tenkan_val < kijun_val) and (ema_12h_val < close_val) and (cloud_thick_val > 0)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price re-enters cloud or Tenkan < Kijun (trend weakening)
            if close_val < cloud_top_val or tenkan_val < kijun_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price re-enters cloud or Tenkan > Kijun (trend weakening)
            if close_val > cloud_bottom_val or tenkan_val > kijun_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_12hTrend_Filter_v2"
timeframe = "6h"
leverage = 1.0