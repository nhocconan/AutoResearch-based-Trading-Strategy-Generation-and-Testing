#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Ichimoku Cloud breakout with 1-week trend filter and volume confirmation.
# Uses Ichimoku's leading span cloud for trend direction and support/resistance.
# Weekly filter ensures trades align with higher timeframe trend.
# Volume confirmation filters out low-participation breakouts.
# Designed for 6h timeframe to target 50-150 trades over 4 years.

name = "6h_ichimoku1w_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-week Ichimoku Cloud components
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    tenkan_sen = np.full(len(close_1w), np.nan)
    # Kijun-sen (Base Line): (26-period high + low)/2
    kijun_sen = np.full(len(close_1w), np.nan)
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2
    senkou_span_a = np.full(len(close_1w), np.nan)
    # Senkou Span B (Leading Span B): (52-period high + low)/2
    senkou_span_b = np.full(len(close_1w), np.nan)
    
    # Calculate Tenkan-sen (9-period)
    for i in range(8, len(close_1w)):
        highest_high = np.max(high_1w[i-8:i+1])
        lowest_low = np.min(low_1w[i-8:i+1])
        tenkan_sen[i] = (highest_high + lowest_low) / 2
    
    # Calculate Kijun-sen (26-period)
    for i in range(25, len(close_1w)):
        highest_high = np.max(high_1w[i-25:i+1])
        lowest_low = np.min(low_1w[i-25:i+1])
        kijun_sen[i] = (highest_high + lowest_low) / 2
    
    # Calculate Senkou Span A
    for i in range(25, len(close_1w)):
        if not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
            senkou_span_a[i] = (tenkan_sen[i] + kijun_sen[i]) / 2
    
    # Calculate Senkou Span B (52-period)
    for i in range(51, len(close_1w)):
        highest_high = np.max(high_1w[i-51:i+1])
        lowest_low = np.min(low_1w[i-51:i+1])
        senkou_span_b[i] = (highest_high + lowest_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen)
    span_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a)
    span_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b)
    
    # 1-day volume average for confirmation
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(vol_1d), np.nan)
    for i in range(19, len(vol_1d)):  # 20-period average
        vol_ma_1d[i] = np.mean(vol_1d[i-19:i+1])
    
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(51, 19)  # Ichimoku needs 52 periods for Senkou B, volume needs 20
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(span_a_aligned[i]) or np.isnan(span_b_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x daily average
        volume_filter = volume[i] > vol_ma_aligned[i] * 1.5
        
        # Cloud top and bottom
        cloud_top = max(span_a_aligned[i], span_b_aligned[i])
        cloud_bottom = min(span_a_aligned[i], span_b_aligned[i])
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price below cloud or stoploss
            if (close[i] < cloud_bottom or 
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above cloud or stoploss
            if (close[i] > cloud_top or 
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries when price breaks cloud with volume
            if volume_filter:
                # Long: price breaks above cloud
                if close[i] > cloud_top and close[i-1] <= cloud_top:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: price breaks below cloud
                elif close[i] < cloud_bottom and close[i-1] >= cloud_bottom:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals