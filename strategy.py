#!/usr/bin/env python3
"""
6h Ichimoku Kumo Breakout with 1d Trend Filter
Hypothesis: Ichimoku system provides robust trend identification and support/resistance.
TK cross above/below cloud with cloud color filter confirms trend strength.
Works in bull (buy when price above cloud + TK cross bullish) and bear (sell when price below cloud + TK cross bearish).
Target: 50-150 trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_kumo_breakout_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for Ichimoku (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    highest_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    lowest_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (highest_tenkan + lowest_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    highest_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    lowest_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (highest_kijun + lowest_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period_senkou = 52
    highest_senkou = pd.Series(high_1d).rolling(window=period_senkou, min_periods=period_senkou).max().values
    lowest_senkou = pd.Series(low_1d).rolling(window=period_senkou, min_periods=period_senkou).min().values
    senkou_b = (highest_senkou + lowest_senkou) / 2
    
    # Align Ichimoku components to 6h
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (max of Ichimoku periods)
    start = 52  # For Senkou B
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or 
            np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Kumo (cloud) boundaries
        upper_kumo = np.maximum(senkou_a_6h[i], senkou_b_6h[i])
        lower_kumo = np.minimum(senkou_a_6h[i], senkou_b_6h[i])
        
        # TK cross
        tk_cross_bullish = tenkan_6h[i] > kijun_6h[i]
        tk_cross_bearish = tenkan_6h[i] < kijun_6h[i]
        
        # Check exits: opposite TK cross or price outside cloud in opposite direction
        if position == 1:  # long position
            # Exit: TK cross bearish OR price below cloud
            if (tk_cross_bearish or close[i] < lower_kumo):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: TK cross bullish OR price above cloud
            if (tk_cross_bullish or close[i] > upper_kumo):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: TK cross + price position relative to cloud + volume
            bull_entry = (tk_cross_bullish and 
                         close[i] > upper_kumo and 
                         volume[i] > vol_ema[i] * 1.5)
            
            bear_entry = (tk_cross_bearish and 
                         close[i] < lower_kumo and 
                         volume[i] > vol_ema[i] * 1.5)
            
            if bull_entry:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_entry:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals