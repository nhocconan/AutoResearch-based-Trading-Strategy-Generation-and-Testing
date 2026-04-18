#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_TF_Trend
Hypothesis: Ichimoku cloud (from daily) + TK cross (6h) + volume confirmation filters false breakouts.
The cloud acts as dynamic support/resistance in both bull/bear markets; TK cross signals momentum.
Uses weekly trend filter to avoid counter-trend trades. Target: 15-30 trades/year (60-120 total).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku components on daily (base period: 9, 26, 52)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 ahead
    senkou_b = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                 pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2)
    # Chikou Span (Lagging Span): close shifted 26 behind
    
    # Align to 6h (Ichimoku values are valid after the daily bar closes)
    tenkan_sen_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_6h = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a.values, additional_delay_bars=26)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b.values, additional_delay_bars=26)
    
    # Weekly trend filter (for direction)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    ema_1w = pd.Series(df_1w['close']).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_1w_6h = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume filter: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Warmup for Ichimoku
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or
            np.isnan(ema_1w_6h[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        tenkan = tenkan_sen_6h[i]
        kijun = kijun_sen_6h[i]
        senkou_a = senkou_a_6h[i]
        senkou_b = senkou_b_6h[i]
        weekly_trend = ema_1w_6h[i]
        vol_ok = volume_filter[i]
        
        # Cloud top and bottom
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        
        if position == 0:
            # Long: TK cross bullish, price above cloud, weekly uptrend, volume
            if tenkan > kijun and price > cloud_top and price > weekly_trend and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: TK cross bearish, price below cloud, weekly downtrend, volume
            elif tenkan < kijun and price < cloud_bottom and price < weekly_trend and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TK cross bearish OR price drops below cloud
            if tenkan < kijun or price < cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TK cross bullish OR price rises above cloud
            if tenkan > kijun or price > cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TF_Trend"
timeframe = "6h"
leverage = 1.0