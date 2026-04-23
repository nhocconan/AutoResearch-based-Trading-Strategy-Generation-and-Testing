#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation
- Uses Ichimoku Cloud (Senkou Span A/B) from 6h timeframe for dynamic support/resistance
- Trend filter: price > 1d EMA(50) for longs, price < 1d EMA(50) for shorts to avoid counter-trend
- Volume confirmation: > 2.0x 20-period average ensures breakout has momentum
- Tenkan/Kijun cross (TK cross) used for entry timing within the breakout zone
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by trading with the 1d trend and using cloud as filter
"""

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
    volume = prices['volume'].values
    
    # Get 6h data for Ichimoku calculations
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    displacement = 26
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_6h).rolling(window=tenkan_period, min_periods=tenkan_period).max() +
                  pd.Series(low_6h).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_6h).rolling(window=kijun_period, min_periods=kijun_period).max() +
                 pd.Series(low_6h).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = (pd.Series(high_6h).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() +
                     pd.Series(low_6h).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2
    
    # Align Ichimoku components to 6h timeframe (no extra displacement needed as align_htf_to_ltf handles it)
    tenkan_aligned = align_htf_to_ltf(prices, df_6h, tenkan_sen.values)
    kijun_aligned = align_htf_to_ltf(prices, df_6h, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_b.values)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(52, 20, 50)  # Senkou Span B, volume MA, 1d EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud boundaries (top and bottom of cloud)
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # TK cross conditions
        tk_cross_up = tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1]
        tk_cross_down = tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1]
        
        # Price relative to cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Trend filter: price > 1d EMA(50) for uptrend, price < 1d EMA(50) for downtrend
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long conditions: price breaks above cloud, TK cross up, uptrend, volume spike
            long_signal = (price_above_cloud and 
                          tk_cross_up and
                          uptrend and
                          volume[i] > 2.0 * vol_ma[i])
            
            # Short conditions: price breaks below cloud, TK cross down, downtrend, volume spike
            short_signal = (price_below_cloud and 
                           tk_cross_down and
                           downtrend and
                           volume[i] > 2.0 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite TK cross or price returns to cloud
            exit_signal = False
            
            if position == 1:
                # Exit long: TK cross down or price falls below cloud bottom
                if (tk_cross_down or 
                    price_below_cloud):
                    exit_signal = True
            elif position == -1:
                # Exit short: TK cross up or price rises above cloud top
                if (tk_cross_up or 
                    price_above_cloud):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_1dEMA50_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0