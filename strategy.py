#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with TK Cross + Volume Spike + Weekly Trend Filter
- Uses Ichimoku cloud (senkou span A/B) from 6h for dynamic support/resistance
- Tenkan-Kijun (TK) cross as momentum signal with confirmation
- Weekly EMA50 trend filter to avoid counter-trend trades in bear markets
- Volume spike (>1.8x 20-period average) to confirm breakout strength
- Discrete position sizing: 0.25 to minimize fee churn
- Target: 12-30 trades/year per symbol (~50-120 total over 4 years)
- Works in bull markets (cloud acts as support in uptrend) and bear markets (cloud acts as resistance in downtrend)
- Ichimoku is proven effective in crypto markets and less saturated than Donchian/Camarilla variants
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
    
    # Get 6h data for Ichimoku calculations
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    displacement = 26
    
    # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past 9 periods
    def calculate_ichimoku_components(high_arr, low_arr, close_arr):
        # Tenkan-sen
        highest_tenkan = pd.Series(high_arr).rolling(window=tenkan_period, min_periods=tenkan_period).max()
        lowest_tenkan = pd.Series(low_arr).rolling(window=tenkan_period, min_periods=tenkan_period).min()
        tenkan = (highest_tenkan + lowest_tenkan) / 2
        
        # Kijun-sen (Base Line): (highest high + lowest low)/2 for past 26 periods
        highest_kijun = pd.Series(high_arr).rolling(window=kijun_period, min_periods=kijun_period).max()
        lowest_kijun = pd.Series(low_arr).rolling(window=kijun_period, min_periods=kijun_period).min()
        kijun = (highest_kijun + lowest_kijun) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
        senkou_a = ((tenkan + kijun) / 2)
        
        # Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past 52 periods shifted 26 periods ahead
        highest_senkou_b = pd.Series(high_arr).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max()
        lowest_senkou_b = pd.Series(low_arr).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()
        senkou_b = ((highest_senkou_b + lowest_senkou_b) / 2)
        
        return tenkan.values, kijun.values, senkou_a.values, senkou_b.values
    
    tenkan_6h, kijun_6h, senkou_a_6h, senkou_b_6h = calculate_ichimoku_components(high_6h, low_6h, close_6h)
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period) on 6h
    volume_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe (primary)
    tenkan_aligned = align_htf_to_ltf(prices, df_6h, tenkan_6h)
    kijun_aligned = align_htf_to_ltf(prices, df_6h, kijun_6h)
    senkou_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_a_6h)
    senkou_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_b_6h)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    volume_ma_aligned = align_htf_to_ltf(prices, df_6h, volume_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for Ichimoku calculations
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        ema_trend = ema50_1w_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        if position == 0:
            # Long conditions:
            # 1. TK cross bullish (tenkan > kijun)
            # 2. Price above cloud (bullish bias)
            # 3. Weekly trend filter: price > weekly EMA50 (uptrend)
            # 4. Volume confirmation: >1.8x average volume
            if (tenkan_val > kijun_val and 
                price > cloud_top and 
                price > ema_trend and 
                vol > 1.8 * vol_ma):
                signals[i] = 0.25
                position = 1
            
            # Short conditions:
            # 1. TK cross bearish (tenkan < kijun)
            # 2. Price below cloud (bearish bias)
            # 3. Weekly trend filter: price < weekly EMA50 (downtrend)
            # 4. Volume confirmation: >1.8x average volume
            elif (tenkan_val < kijun_val and 
                  price < cloud_bottom and 
                  price < ema_trend and 
                  vol > 1.8 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when:
            # 1. TK cross turns bearish (tenkan < kijun) OR
            # 2. Price falls below cloud bottom
            if tenkan_val < kijun_val or price < cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when:
            # 1. TK cross turns bullish (tenkan > kijun) OR
            # 2. Price rises above cloud top
            if tenkan_val > kijun_val or price > cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_IchimokuTKCross_CloudFilter_WeeklyEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0