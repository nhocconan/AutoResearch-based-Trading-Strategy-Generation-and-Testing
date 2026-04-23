#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1d ADX regime filter and volume confirmation
- Uses Ichimoku components (Tenkan-sen, Kijun-sen, Senkou Span A/B) from 6h timeframe
- 1d ADX > 25 defines trending regime: only trade in strong trends
- Tenkan/Kijun cross provides entry signals with cloud as dynamic support/resistance
- Volume confirmation (> 1.5x 20-period average) filters false breakouts
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by only trading when ADX confirms strong trend
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
    
    # Calculate 6h Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    highest_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    lowest_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (highest_tenkan + lowest_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    highest_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    lowest_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (highest_kijun + lowest_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    highest_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    lowest_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = (highest_senkou_b + lowest_senkou_b) / 2
    
    # Calculate 1d ADX for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = np.diff(low_1d, prepend=low_1d[0]) * -1  # Invert to positive
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    period_adx = 14
    alpha = 1.0 / period_adx
    
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (1 - alpha) * atr[i-1] + alpha * tr[i]
    
    plus_di = np.zeros_like(plus_dm)
    minus_di = np.zeros_like(minus_dm)
    
    # Avoid division by zero
    atr_safe = np.where(atr == 0, 1e-10, atr)
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean().values / atr_safe
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean().values / atr_safe
    
    dx = np.zeros_like(plus_di)
    dx_denom = plus_di + minus_di
    dx_denom_safe = np.where(dx_denom == 0, 1e-10, dx_denom)
    dx = 100 * np.abs(plus_di - minus_di) / dx_denom_safe
    
    adx = pd.Series(dx).ewm(alpha=alpha, adjust=False).mean().values
    
    # Align HTF data to LTF
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(52, 20)  # for Senkou Span B and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(span_a_aligned[i]) or np.isnan(span_b_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(span_a_aligned[i], span_b_aligned[i])
        cloud_bottom = min(span_a_aligned[i], span_b_aligned[i])
        
        if position == 0:
            # Long conditions: Tenkan crosses above Kijun, price above cloud, ADX > 25, volume confirmation
            tenkan_prev = tenkan_aligned[i-1] if i > 0 else tenkan_aligned[i]
            kijun_prev = kijun_aligned[i-1] if i > 0 else kijun_aligned[i]
            
            tenkan_cross_above = tenkan_aligned[i] > kijun_aligned[i] and tenkan_prev <= kijun_prev
            price_above_cloud = close[i] > cloud_top
            strong_trend = adx_aligned[i] > 25
            volume_confirmed = volume[i] > 1.5 * vol_ma[i]
            
            if tenkan_cross_above and price_above_cloud and strong_trend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            
            # Short conditions: Tenkan crosses below Kijun, price below cloud, ADX > 25, volume confirmation
            tenkan_cross_below = tenkan_aligned[i] < kijun_aligned[i] and tenkan_prev >= kijun_prev
            price_below_cloud = close[i] < cloud_bottom
            
            if tenkan_cross_below and price_below_cloud and strong_trend and volume_confirmed:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Tenkan/Kijun cross or price returns to cloud
            exit_signal = False
            
            if position == 1:
                # Exit long: Tenkan crosses below Kijun or price falls below cloud bottom
                tenkan_prev = tenkan_aligned[i-1] if i > 0 else tenkan_aligned[i]
                kijun_prev = kijun_aligned[i-1] if i > 0 else kijun_aligned[i]
                tenkan_cross_below = tenkan_aligned[i] < kijun_aligned[i] and tenkan_prev >= kijun_prev
                price_below_cloud = close[i] < cloud_bottom
                
                if tenkan_cross_below or price_below_cloud:
                    exit_signal = True
            elif position == -1:
                # Exit short: Tenkan crosses above Kijun or price rises above cloud top
                tenkan_prev = tenkan_aligned[i-1] if i > 0 else tenkan_aligned[i]
                kijun_prev = kijun_aligned[i-1] if i > 0 else kijun_aligned[i]
                tenkan_cross_above = tenkan_aligned[i] > kijun_aligned[i] and tenkan_prev <= kijun_prev
                price_above_cloud = close[i] > cloud_top
                
                if tenkan_cross_above or price_above_cloud:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_ADX25_VolumeConfirm"
timeframe = "6h"
leverage = 1.0