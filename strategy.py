#!/usr/bin/env python3
"""
6h_ADX_Ichimoku_Cloud_Strategy_v1
Hypothesis: Ichimoku cloud acts as dynamic support/resistance in 6h timeframe, while ADX filters for trending conditions.
In bull markets, price stays above cloud with ADX>25; in bear markets, price stays below cloud with ADX>25.
Cloud twist (Tenkan/Kijun cross) provides entry signals with trend confirmation.
Designed for low trade frequency (15-35/year) to minimize fee drag on 6h timeframe.
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
    
    # Get daily data for Ichimoku cloud (calculated once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Ichimoku components (9, 26, 52 periods)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 52 periods ahead
    senkou_b = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                 pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2).shift(52)
    
    # Align Ichimoku components to 6h timeframe (wait for close of daily bar)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a.values)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b.values)
    
    # ADX for trend strength (14-period) on 6h data
    # Calculate True Range
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smooth TR and DM
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm14_plus = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm14_minus = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus = 100 * dm14_plus / tr14
    di_minus = 100 * dm14_minus / tr14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Initialize arrays
    tenkan_sen_aligned = np.where(np.isnan(tenkan_sen_aligned), 0, tenkan_sen_aligned)
    kijun_sen_aligned = np.where(np.isnan(kijun_sen_aligned), 0, kijun_sen_aligned)
    senkou_a_aligned = np.where(np.isnan(senkou_a_aligned), 0, senkou_a_aligned)
    senkou_b_aligned = np.where(np.isnan(senkou_b_aligned), 0, senkou_b_aligned)
    adx = np.where(np.isnan(adx), 0, adx)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical value is invalid
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        tenkan = tenkan_sen_aligned[i]
        kijun = kijun_sen_aligned[i]
        senkou_a = senkou_a_aligned[i]
        senkou_b = senkou_b_aligned[i]
        adx_val = adx[i]
        
        # Cloud boundaries (Senkou Span A and B)
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        
        # Check if price is above or below cloud
        price_above_cloud = price > cloud_top
        price_below_cloud = price < cloud_bottom
        
        # TK Cross signals
        tk_cross_up = tenkan > kijun and tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1]
        tk_cross_down = tenkan < kijun and tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1]
        
        if position == 0:
            # Long: TK cross up, price above cloud, strong trend (ADX > 25)
            if tk_cross_up and price_above_cloud and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: TK cross down, price below cloud, strong trend (ADX > 25)
            elif tk_cross_down and price_below_cloud and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: TK cross down OR price falls below cloud
            if tk_cross_down or price < cloud_top:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: TK cross up OR price rises above cloud
            if tk_cross_up or price > cloud_bottom:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ADX_Ichimoku_Cloud_Strategy_v1"
timeframe = "6h"
leverage = 1.0