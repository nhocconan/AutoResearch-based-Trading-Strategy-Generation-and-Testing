#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h trend following using 12h Ichimoku cloud with volume confirmation.
# Long: price above 12h Ichimoku cloud (Senkou Span A/B) + Tenkan > Kijun + volume > 1.5x avg volume
# Short: price below 12h Ichimoku cloud + Tenkan < Kijun + volume > 1.5x avg volume
# Ichimoku calculated from 12h data: Tenkan = (9-period high + low)/2, Kijun = (26-period high + low)/2
# Senkou Span A = (Tenkan + Kijun)/2 shifted 26 periods ahead, Senkou Span B = (52-period high + low)/2 shifted 52 periods ahead
# Cloud acts as dynamic support/resistance; price above/below cloud indicates trend direction
# Volume confirmation reduces false breakouts
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Works in both bull and bear markets by using 12h Ichimoku as trend filter

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12-hour data for Ichimoku
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 52:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Ichimoku components
    # Tenkan-sen (Conversion Line): 9-period high-low midpoint
    tenkan = np.full(len(close_12h), np.nan)
    for i in range(8, len(close_12h)):
        tenkan[i] = (np.max(high_12h[i-8:i+1]) + np.min(low_12h[i-8:i+1])) / 2
    
    # Kijun-sen (Base Line): 26-period high-low midpoint
    kijun = np.full(len(close_12h), np.nan)
    for i in range(25, len(close_12h)):
        kijun[i] = (np.max(high_12h[i-25:i+1]) + np.min(low_12h[i-25:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = np.full(len(close_12h), np.nan)
    for i in range(len(tenkan)):
        if not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
            senkou_a[i] = (tenkan[i] + kijun[i]) / 2
    # Shift 26 periods ahead (for plotting, but we need current values)
    # For current cloud, we use unshifted Senkou Span A and B
    
    # Senkou Span B (Leading Span B): 52-period high-low midpoint shifted 52 periods ahead
    senkou_b = np.full(len(close_12h), np.nan)
    for i in range(51, len(close_12h)):
        senkou_b[i] = (np.max(high_12h[i-51:i+1]) + np.min(low_12h[i-51:i+1])) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_12h, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_12h, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_12h, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_12h, senkou_b)
    
    # Average volume (20-period = 20*6h = 5 days) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(52, n):
        # Skip if any required data is not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        
        # Determine cloud boundaries (higher of Senkou A/B for resistance, lower for support)
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: price above cloud + Tenkan > Kijun + volume confirmation
            if (price > cloud_top and 
                tenkan_val > kijun_val and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: price below cloud + Tenkan < Kijun + volume confirmation
            elif (price < cloud_bottom and 
                  tenkan_val < kijun_val and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls below cloud or Tenkan < Kijun
            if (price < cloud_bottom or
                tenkan_val < kijun_val):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price rises above cloud or Tenkan > Kijun
            if (price > cloud_top or
                tenkan_val > kijun_val):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_12h_Ichimoku_Cloud_Volume"
timeframe = "6h"
leverage = 1.0