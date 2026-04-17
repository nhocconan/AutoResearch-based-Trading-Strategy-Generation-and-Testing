#!/usr/bin/env python3
"""
12h_1d_Ichimoku_Bounce_Trend
Strategy: 12-hour Ichimoku Cloud bounce with trend confirmation.
Long: Price touches Kumo (cloud) support + Tenkan > Kijun + price above daily EMA50
Short: Price touches Kumo resistance + Tenkan < Kijun + price below daily EMA50
Exit: Price crosses opposite Kumo boundary or Tenkan/Kijun cross reverses
Position size: 0.25
Designed to capture trend continuations after pullbacks in both bull and bear markets.
Timeframe: 12h
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Ichimoku components on 12h data
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = ((tenkan + kijun) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2.0)
    
    # Kumo (Cloud) boundaries: Senkou Span A and B shifted forward 26 periods
    # For backtesting, we use current Senkou spans as cloud boundaries
    # Cloud top = max(Senkou A, Senkou B), Cloud bottom = min(Senkou A, Senkou B)
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    close_series_1d = pd.Series(close_1d)
    ema50_1d = close_series_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Need enough data for Senkou B
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close[i] > ema50_1d_aligned[i]
        price_below_ema = close[i] < ema50_1d_aligned[i]
        
        # Ichimoku signals
        price_in_cloud = (close[i] >= cloud_bottom[i]) and (close[i] <= cloud_top[i])
        price_above_cloud = close[i] > cloud_top[i]
        price_below_cloud = close[i] < cloud_bottom[i]
        tenkan_above_kijun = tenkan[i] > kijun[i]
        tenkan_below_kijun = tenkan[i] < kijun[i]
        
        # Touch cloud boundaries (within 0.3% of cloud edge)
        touch_cloud_top = abs(close[i] - cloud_top[i]) < (0.003 * cloud_top[i])
        touch_cloud_bottom = abs(close[i] - cloud_bottom[i]) < (0.003 * cloud_bottom[i])
        
        if position == 0:
            # Long: touch cloud bottom + Tenkan > Kijun + price above daily EMA
            if touch_cloud_bottom and tenkan_above_kijun and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short: touch cloud top + Tenkan < Kijun + price below daily EMA
            elif touch_cloud_top and tenkan_below_kijun and price_below_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below cloud bottom OR Tenkan < Kijun
            if close[i] < cloud_bottom[i] or tenkan_below_kijun:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above cloud top OR Tenkan > Kijun
            if close[i] > cloud_top[i] or tenkan_above_kijun:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Ichimoku_Bounce_Trend"
timeframe = "12h"
leverage = 1.0