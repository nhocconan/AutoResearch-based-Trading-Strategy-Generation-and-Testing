#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filter_1dTrend
Hypothesis: Ichimoku Cloud (Tenkan/Kijun cross) on 6h combined with 1d trend filter and volume confirmation
provides high-probability entries in both bull and bear markets. Cloud acts as dynamic support/resistance,
reducing whipsaws. Targets 50-150 total trades over 4 years for low fee drag.
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
    volume = prices['volume'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    tenkan_sen = (pd.Series(high).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + low) / 2
    kijun_sen = (pd.Series(high).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    senkou_span_b = (pd.Series(high).rolling(window=52, min_periods=52).max() + 
                     pd.Series(low).rolling(window=52, min_periods=52).min()) / 2
    
    # 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # Align 1d indicators to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm)
    
    # Align Senkou Span B (needs 26-period displacement)
    senkou_span_b_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), senkou_span_b.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need Ichimoku (52), EMA50 (50), volume avg (20)
    start_idx = max(52, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen.iloc[i]) or np.isnan(kijun_sen.iloc[i]) or 
            np.isnan(senkou_span_a.iloc[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        tenkan = tenkan_sen.iloc[i]
        kijun = kijun_sen.iloc[i]
        span_a = senkou_span_a.iloc[i]
        span_b = senkou_span_b_aligned[i]
        ema50 = ema50_1d_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        
        # Cloud top and bottom
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        if position == 0:
            # Bullish: Tenkan > Kijun AND price above cloud AND 1d uptrend
            bullish_cross = tenkan > kijun
            price_above_cloud = close[i] > cloud_top
            uptrend = close[i] > ema50
            
            if bullish_cross and price_above_cloud and uptrend and vol_conf:
                signals[i] = size
                position = 1
            
            # Bearish: Tenkan < Kijun AND price below cloud AND 1d downtrend
            bearish_cross = tenkan < kijun
            price_below_cloud = close[i] < cloud_bottom
            downtrend = close[i] < ema50
            
            if bearish_cross and price_below_cloud and downtrend and vol_conf:
                signals[i] = -size
                position = -1
        
        elif position == 1:
            # Exit: Tenkan < Kijun OR price below cloud bottom
            if tenkan < kijun or close[i] < cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        
        elif position == -1:
            # Exit: Tenkan > Kijun OR price above cloud top
            if tenkan > kijun or close[i] > cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Cloud_Filter_1dTrend"
timeframe = "6h"
leverage = 1.0