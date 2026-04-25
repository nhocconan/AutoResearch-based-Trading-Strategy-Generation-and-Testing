#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_VolumeSpike_Trend
Hypothesis: Trade 6h Ichimoku cloud twists (Senkou Span A/B cross) with 1d trend filter (price >/= EMA50) and volume confirmation (>2.0x 20-bar MA). 
Ichimoku cloud twist signals strong momentum shifts. 1d EMA50 ensures trading with higher timeframe trend. Volume confirmation adds conviction. 
Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag. Discrete sizing 0.25 balances profit and fee drag. 
Works in bull/bear: trend filter adapts to market direction, volume confirms breakout validity.
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
    
    # Get 6h data for Ichimoku calculation
    df_6h = get_htf_data(prices, '6h')
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
    tenkan_sen_values = tenkan_sen.values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_6h).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                 pd.Series(low_6h).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    kijun_sen_values = kijun_sen.values
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    senkou_span_a_values = senkou_span_a.values
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = (pd.Series(high_6h).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                     pd.Series(low_6h).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2
    senkou_span_b_values = senkou_span_b.values
    
    # Align Ichimoku components to primary 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_6h, tenkan_sen_values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_6h, kijun_sen_values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_a_values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_b_values)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Ichimoku (52) and 1d EMA50 (50) and volume MA (20)
    start_idx = max(52, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Bullish twist: Senkou Span A crosses above Senkou Span B AND price > 1d EMA50 AND volume confirm
            bullish_twist = (senkou_span_a_aligned[i] > senkou_span_b_aligned[i]) and \
                            (senkou_span_a_aligned[i-1] <= senkou_span_b_aligned[i-1]) and \
                            (close[i] > ema_50_1d_aligned[i]) and \
                            volume_confirm[i]
            # Bearish twist: Senkou Span A crosses below Senkou Span B AND price < 1d EMA50 AND volume confirm
            bearish_twist = (senkou_span_a_aligned[i] < senkou_span_b_aligned[i]) and \
                            (senkou_span_a_aligned[i-1] >= senkou_span_b_aligned[i-1]) and \
                            (close[i] < ema_50_1d_aligned[i]) and \
                            volume_confirm[i]
            
            if bullish_twist:
                signals[i] = 0.25
                position = 1
            elif bearish_twist:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: Senkou Span A crosses below Senkou Span B OR price < 1d EMA50
            if (senkou_span_a_aligned[i] < senkou_span_b_aligned[i]) or \
               (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: Senkou Span A crosses above Senkou Span B OR price > 1d EMA50
            if (senkou_span_a_aligned[i] > senkou_span_b_aligned[i]) or \
               (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_VolumeSpike_Trend"
timeframe = "6h"
leverage = 1.0