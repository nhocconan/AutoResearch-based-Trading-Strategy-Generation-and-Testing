#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_Trend
Hypothesis: Use Ichimoku cloud from 1d timeframe for trend direction and 6h for entry timing.
Long when price breaks above Kumo cloud (Senkou Span B) with Tenkan > Kijun.
Short when price breaks below Kumo cloud with Tenkan < Kijun.
Requires volume > 1.5x 20-period average for confirmation.
Uses ADX > 20 on 1d to filter for trending conditions only.
Ichimoku works in all markets as it combines momentum, trend, and support/resistance.
Target: 20-50 trades/year by requiring multiple confirmations.
Works in bull/bear markets by only taking trades in direction of higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
    """Calculate Ichimoku components"""
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past 9 periods
    tenkan_sen = (pd.Series(high).rolling(window=tenkan, min_periods=tenkan).max() + 
                  pd.Series(low).rolling(window=tenkan, min_periods=tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 for past 26 periods
    kijun_sen = (pd.Series(high).rolling(window=kijun, min_periods=kijun).max() + 
                 pd.Series(low).rolling(window=kijun, min_periods=kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past 52 periods shifted 26 ahead
    senkou_span_b = (pd.Series(high).rolling(window=senkou, min_periods=senkou).max() + 
                     pd.Series(low).rolling(window=senkou, min_periods=senkou).min()) / 2
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind (not used for signals)
    
    return tenkan_sen.values, kijun_sen.values, senkou_span_a.values, senkou_span_b.values

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth using Wilder's smoothing
    tr_period = np.zeros_like(tr)
    dm_plus_period = np.zeros_like(dm_plus)
    dm_minus_period = np.zeros_like(dm_minus)
    
    tr_period[0] = tr[0]
    dm_plus_period[0] = dm_plus[0]
    dm_minus_period[0] = dm_minus[0]
    
    for i in range(1, len(tr)):
        tr_period[i] = tr_period[i-1] - (tr_period[i-1] / period) + tr[i]
        dm_plus_period[i] = dm_plus_period[i-1] - (dm_plus_period[i-1] / period) + dm_plus[i]
        dm_minus_period[i] = dm_minus_period[i-1] - (dm_minus_period[i-1] / period) + dm_minus[i]
    
    # Calculate DI+ and DI-
    di_plus = np.where(tr_period != 0, 100 * dm_plus_period / tr_period, 0)
    di_minus = np.where(tr_period != 0, 100 * dm_minus_period / tr_period, 0)
    
    # Calculate DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    
    adx = np.zeros_like(dx)
    if len(dx) >= period:
        adx[period-1] = np.mean(dx[:period])  # First ADX value
        
        for i in range(period, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data once for Ichimoku and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Calculate daily Ichimoku
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tenkan, kijun, senkou_a, senkou_b = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Kumo cloud top and bottom (Senkou Span B is the slower one)
    kumo_top = np.maximum(senkou_a, senkou_b)
    kumo_bottom = np.minimum(senkou_a, senkou_b)
    
    # Calculate daily ADX for trend filter
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align all indicators to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    kumo_top_6h = align_htf_to_ltf(prices, df_1d, kumo_top)
    kumo_bottom_6h = align_htf_to_ltf(prices, df_1d, kumo_bottom)
    adx_1d_6h = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or np.isnan(kumo_top_6h[i]) or np.isnan(kumo_bottom_6h[i]) or np.isnan(adx_1d_6h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        # Trend filter: ADX > 20 indicates trending market
        trending = adx_1d_6h[i] > 20
        
        if position == 0:
            # Long: price above cloud AND Tenkan > Kijun AND volume + trend confirmation
            if price > kumo_top_6h[i] and tenkan_6h[i] > kijun_6h[i] and volume_ok and trending:
                signals[i] = 0.25
                position = 1
            # Short: price below cloud AND Tenkan < Kijun AND volume + trend confirmation
            elif price < kumo_bottom_6h[i] and tenkan_6h[i] < kijun_6h[i] and volume_ok and trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price drops below cloud base OR Tenkan < Kijun (momentum loss)
            if price < kumo_bottom_6h[i] or tenkan_6h[i] < kijun_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above cloud top OR Tenkan > Kijun (momentum loss)
            if price > kumo_top_6h[i] or tenkan_6h[i] > kijun_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_Trend"
timeframe = "6h"
leverage = 1.0