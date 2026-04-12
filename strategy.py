#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 periods for Ichimoku
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components (Tenkan-sen, Kijun-sen, Senkou Span A/B, Chikou Span)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_52 + low_52) / 2)
    
    # Align Ichimoku components to 6h timeframe (with proper look-ahead prevention)
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Volume filter: 20-period EMA on 6b volume
    vol_ema = np.full(n, np.nan)
    vol_series = pd.Series(volume)
    vol_ema_values = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema[:] = vol_ema_values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or 
            np.isnan(vol_ema[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x EMA
        volume_filter = volume[i] > vol_ema[i] * 1.3
        
        # Cloud condition: price above/below cloud
        cloud_top = np.maximum(senkou_a_6h[i], senkou_b_6h[i])
        cloud_bottom = np.minimum(senkou_a_6h[i], senkou_b_6h[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # TK Cross conditions
        tk_cross_bull = tenkan_6h[i] > kijun_6h[i]
        tk_cross_bear = tenkan_6h[i] < kijun_6h[i]
        
        # Entry conditions: TK cross with cloud filter and volume
        long_entry = tk_cross_bull and price_above_cloud and volume_filter
        short_entry = tk_cross_bear and price_below_cloud and volume_filter
        
        # Exit conditions: Opposite TK cross or price enters cloud
        tk_cross_exit_bull = tenkan_6h[i] < kijun_6h[i]  # Bearish cross exits long
        tk_cross_exit_bear = tenkan_6h[i] > kijun_6h[i]  # Bullish cross exits short
        price_in_cloud = (close[i] >= cloud_bottom) and (close[i] <= cloud_top)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and (tk_cross_exit_bull or price_in_cloud):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (tk_cross_exit_bear or price_in_cloud):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_ichimoku_tk_cross_volume_filter_v1"
timeframe = "6h"
leverage = 1.0