#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Tenkan_Kijun_Cross_v1
6-hour strategy using Ichimoku Cloud from 1d with Tenkan-Kijun cross from 6h.
Enters long when price above cloud + Tenkan > Kijun, short when price below cloud + Tenkan < Kijun.
Exits when price crosses Tenkan-Kijun in opposite direction.
Uses 12h ADX to avoid ranging markets (ADX < 20).
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # === Ichimoku Cloud from Daily (1d) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    tenkan_sen_1d = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                     pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + low)/2
    kijun_sen_1d = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                    pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a_1d = ((tenkan_sen_1d + kijun_sen_1d) / 2)
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    senkou_span_b_1d = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                         pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2)
    
    # Align Ichimoku components to 6h timeframe (wait for daily close)
    tenkan_sen_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen_1d.values)
    kijun_sen_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen_1d.values)
    senkou_span_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d.values)
    senkou_span_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d.values)
    
    # === 6h Tenkan-Kijun Cross for Entry Timing ===
    # Tenkan-sen (6h): (9-period high + low)/2
    tenkan_sen_6h = (pd.Series(high).rolling(window=9, min_periods=9).max() + 
                     pd.Series(low).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (6h): (26-period high + low)/2
    kijun_sen_6h = (pd.Series(high).rolling(window=26, min_periods=26).max() + 
                    pd.Series(low).rolling(window=26, min_periods=26).min()) / 2
    
    # === 12h ADX for Regime Filter (avoid ranging) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX
    plus_dm = np.zeros_like(high_12h)
    minus_dm = np.zeros_like(low_12h)
    plus_dm[1:] = np.maximum(high_12h[1:] - high_12h[:-1], 0)
    minus_dm[1:] = np.maximum(low_12h[:-1] - low_12h[1:], 0)
    plus_dm = np.where(plus_dm > minus_dm, plus_dm, 0)
    minus_dm = np.where(minus_dm > plus_dm, minus_dm, 0)
    
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = high_12h[0] - low_12h[0]
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / (atr * 14)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / (atr * 14)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_1d_aligned[i]) or 
            np.isnan(kijun_sen_1d_aligned[i]) or 
            np.isnan(senkou_span_a_1d_aligned[i]) or 
            np.isnan(senkou_span_b_1d_aligned[i]) or 
            np.isnan(tenkan_sen_6h[i]) or 
            np.isnan(kijun_sen_6h[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Determine cloud boundaries (Senkou Span A/B)
        upper_cloud = np.maximum(senkou_span_a_1d_aligned[i], senkou_span_b_1d_aligned[i])
        lower_cloud = np.minimum(senkou_span_a_1d_aligned[i], senkou_span_b_1d_aligned[i])
        
        # Trend filter: only trade when ADX > 20 (trending market)
        trending = adx_aligned[i] > 20
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price above cloud + Tenkan > Kijun (bullish alignment)
            if (close[i] > upper_cloud and 
                tenkan_sen_6h[i] > kijun_sen_6h[i] and 
                trending):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price below cloud + Tenkan < Kijun (bearish alignment)
            elif (close[i] < lower_cloud and 
                  tenkan_sen_6h[i] < kijun_sen_6h[i] and 
                  trending):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: Tenkan-Kijun cross in opposite direction
        elif position == 1:
            # Exit long: Tenkan crosses below Kijun
            if tenkan_sen_6h[i] < kijun_sen_6h[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Tenkan crosses above Kijun
            if tenkan_sen_6h[i] > kijun_sen_6h[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Tenkan_Kijun_Cross_v1"
timeframe = "6h"
leverage = 1.0