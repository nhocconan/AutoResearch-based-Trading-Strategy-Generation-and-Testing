# Solution
#!/usr/bin/env python3
"""
6h Ichimoku Cloud with 1d Trend Filter and Volume Spike
Hypothesis: Ichimoku provides robust trend identification (TK cross + cloud position).
Combined with 1d trend filter (price above/below Kumo) and volume spikes,
it captures high-probability trend continuations in both bull and bear markets.
Tenkan/Kijun cross gives entry timing, cloud acts as dynamic support/resistance.
Low trade frequency due to multiple confluence requirements.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
    """Calculate Ichimoku Cloud components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (np.maximum.accumulate(high)[tenkan-1:] + np.minimum.accumulate(low)[tenkan-1:]) / 2
    # Pad beginning with NaN
    tenkan_sen = np.concatenate([np.full(tenkan-1, np.nan), tenkan_sen])
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (np.maximum.accumulate(high)[kijun-1:] + np.minimum.accumulate(low)[kijun-1:]) / 2
    kijun_sen = np.concatenate([np.full(kijun-1, np.nan), kijun_sen])
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    senkou_a = np.roll(senkou_a, -kijun)  # Shift forward
    senkou_a[:kijun] = np.nan  # First kijun values invalid
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b_high = np.maximum.accumulate(high)[senkou-1:]
    senkou_b_low = np.minimum.accumulate(low)[senkou-1:]
    senkou_b = (senkou_b_high + senkou_b_low) / 2
    senkou_b = np.concatenate([np.full(senkou-1, np.nan), senkou_b])
    senkou_b = np.roll(senkou_b, -kijun)  # Shift forward
    senkou_b[:kijun+senkou] = np.nan  # First kijun+senkou values invalid
    
    # Chikou Span (Lagging Span): close shifted 26 periods behind
    chikou_span = np.roll(close, kijun)  # Shift backward
    chikou_span[-kijun:] = np.nan  # Last kijun values invalid
    
    return tenkan_sen, kijun_sen, senkou_a, senkou_b, chikou_span

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate Ichimoku on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d, chikou_1d = calculate_ichimoku(
        high_1d, low_1d, close_1d
    )
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    chikou_aligned = align_htf_to_ltf(prices, df_1d, chikou_1d)
    
    # Volume spike: current volume > 2.5x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1])
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 2.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Warmup for Ichimoku
    
    for i in range(start_idx, n):
        # Skip if any Ichimoku values are NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i])):
            signals[i] = 0.0
            continue
        
        tenkan = tenkan_aligned[i]
        kijun = kijun_aligned[i]
        senkou_a = senkou_a_aligned[i]
        senkou_b = senkou_b_aligned[i]
        chikou = chikou_aligned[i]
        vol_ok = vol_spike[i]
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        
        # Determine trend: price above/below cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # TK cross signals
        tk_cross_bull = tenkan > kijun
        tk_cross_bear = tenkan < kijun
        
        if position == 0:
            # Enter long: price above cloud + TK bullish cross + chikou above price (26 periods ago) + volume spike
            if (price_above_cloud and tk_cross_bull and 
                not np.isnan(chikou) and chikou > close[i] and vol_ok):
                signals[i] = 0.25
                position = 1
            # Enter short: price below cloud + TK bearish cross + chikou below price + volume spike
            elif (price_below_cloud and tk_cross_bear and 
                  not np.isnan(chikou) and chikou < close[i] and vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below cloud OR TK bearish cross
            if price_below_cloud or tk_cross_bear:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above cloud OR TK bullish cross
            if price_above_cloud or tk_cross_bull:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_VolumeSpike_TrendFilter"
timeframe = "6h"
leverage = 1.0