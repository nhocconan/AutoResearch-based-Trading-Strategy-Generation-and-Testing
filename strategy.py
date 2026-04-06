#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14095_6d_ichimoku_cloud_tf"
timeframe = "6h"
leverage = 1.0

def calculate_ema(arr, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(arr).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for Ichimoku cloud and trend (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    high_52 = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_52 + low_52) / 2)
    
    # Align Ichimoku components to 6h timeframe (shifted by 1 for completed bars)
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (max of 52 for Ichimoku, 20 for volume)
    start = max(52, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or np.isnan(senkou_a_6h[i]) or \
           np.isnan(senkou_b_6h[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Ichimoku signals
        # Bullish: Tenkan > Kijun and price above cloud
        # Bearish: Tenkan < Kijun and price below cloud
        bullish = tenkan_6h[i] > kijun_6h[i] and close[i] > max(senkou_a_6h[i], senkou_b_6h[i])
        bearish = tenkan_6h[i] < kijun_6h[i] and close[i] < min(senkou_a_6h[i], senkou_b_6h[i])
        
        # Generate signals with volume confirmation
        if position == 0:
            if bullish and vol_filter[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bearish and vol_filter[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on bearish cross or price below cloud
            if not bullish or close[i] < min(senkou_a_6h[i], senkou_b_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on bullish cross or price above cloud
            if not bearish or close[i] > max(senkou_a_6h[i], senkou_b_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals