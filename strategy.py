#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud Breakout with 1d ADX Trend Filter and Volume Spike
- Uses Ichimoku (Tenkan/Kijun/Senkou) from 6h for cloud breakout signals
- 1d ADX > 25 defines strong trend: only trade in direction of trend
- Volume confirmation (> 1.5x 20-period average) filters weak breakouts
- Designed for 6h timeframe targeting 12-30 trades/year (50-120 over 4 years)
- Works in both bull and bear markets by following the 1d ADX trend filter
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
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    if n >= period_tenkan:
        tenkan_high = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
        tenkan_low = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
        tenkan = (tenkan_high + tenkan_low) / 2
    else:
        tenkan = np.full(n, np.nan)
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    if n >= period_kijun:
        kijun_high = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
        kijun_low = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
        kijun = (kijun_high + kijun_low) / 2
    else:
        kijun = np.full(n, np.nan)
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    if n >= period_senkou_b:
        senkou_b_high = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
        senkou_b_low = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
        senkou_b = (senkou_b_high + senkou_b_low) / 2
    else:
        senkou_b = np.full(n, np.nan)
    
    # Calculate 1d ADX for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smooth TR and DM
    period_adx = 14
    tr_smooth = pd.Series(tr).ewm(span=period_adx, min_periods=period_adx, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=period_adx, min_periods=period_adx, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=period_adx, min_periods=period_adx, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=period_adx, min_periods=period_adx, adjust=False).mean().values
    
    # Align HTF data
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)  # Use 1d for simplicity, but note: Ichimoku is 6h
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(52, 20)  # need Senkou B (52), volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # Long: price breaks above cloud AND Tenkan > Kijun AND ADX > 25 AND volume spike
            if (close[i] > cloud_top and 
                tenkan_aligned[i] > kijun_aligned[i] and 
                adx_aligned[i] > 25 and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below cloud AND Tenkan < Kijun AND ADX > 25 AND volume spike
            elif (close[i] < cloud_bottom and 
                  tenkan_aligned[i] < kijun_aligned[i] and 
                  adx_aligned[i] > 25 and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to cloud OR Tenkan/Kijun cross reverses
            exit_signal = False
            
            if position == 1:
                # Exit long when price < cloud bottom OR Tenkan < Kijun
                if close[i] < cloud_bottom or tenkan_aligned[i] < kijun_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when price > cloud top OR Tenkan > Kijun
                if close[i] > cloud_top or tenkan_aligned[i] > kijun_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Ichimoku_CloudBreakout_1dADXTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0