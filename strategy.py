#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d timeframe with 1-week Ichimoku Cloud and volume confirmation.
# Long: Price above Kumo (cloud) + Tenkan > Kijun + volume > 1.5x average volume (20-period).
# Short: Price below Kumo + Tenkan < Kijun + volume > 1.5x average volume.
# Uses weekly Ichimoku for trend structure and 1d for execution with volume confirmation.
# Works in bull markets (trend following) and bear markets (counter-trend via cloud rejection).
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w data for Ichimoku Cloud
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:  # Need at least 52 weeks for proper calculation
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Ichimoku components (standard parameters)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    # Chikou Span (Lagging Span): Close plotted 26 periods behind
    
    # Calculate Tenkan-sen (9-period)
    tenkan_sen = np.full(len(close_1w), np.nan)
    for i in range(8, len(close_1w)):
        period_high = np.max(high_1w[i-8:i+1])
        period_low = np.min(low_1w[i-8:i+1])
        tenkan_sen[i] = (period_high + period_low) / 2
    
    # Calculate Kijun-sen (26-period)
    kijun_sen = np.full(len(close_1w), np.nan)
    for i in range(25, len(close_1w)):
        period_high = np.max(high_1w[i-25:i+1])
        period_low = np.min(low_1w[i-25:i+1])
        kijun_sen[i] = (period_high + period_low) / 2
    
    # Calculate Senkou Span A
    senkou_span_a = np.full(len(close_1w), np.nan)
    for i in range(25, len(close_1w)):
        if not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
            senkou_span_a[i] = (tenkan_sen[i] + kijun_sen[i]) / 2
    
    # Calculate Senkou Span B (52-period)
    senkou_span_b = np.full(len(close_1w), np.nan)
    for i in range(51, len(close_1w)):
        period_high = np.max(high_1w[i-51:i+1])
        period_low = np.min(low_1w[i-51:i+1])
        senkou_span_b[i] = (period_high + period_low) / 2
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align 1w Ichimoku components to 1d
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if any required data is not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        tenkan = tenkan_sen_aligned[i]
        kijun = kijun_sen_aligned[i]
        span_a = senkou_span_a_aligned[i]
        span_b = senkou_span_b_aligned[i]
        
        # Kumo (Cloud) boundaries
        upper_kumo = max(span_a, span_b)
        lower_kumo = min(span_a, span_b)
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: Price above Kumo + Tenkan > Kijun + volume confirmation
            if (price > upper_kumo and tenkan > kijun and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Price below Kumo + Tenkan < Kijun + volume confirmation
            elif (price < lower_kumo and tenkan < kijun and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price closes below Kumo (cloud)
            if price < lower_kumo:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Price closes above Kumo (cloud)
            if price > upper_kumo:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Ichimoku_Cloud_Volume"
timeframe = "1d"
leverage = 1.0