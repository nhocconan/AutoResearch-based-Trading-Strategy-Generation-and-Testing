#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Ichimoku_Kumo_Twist_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # Chikou Span (Lagging Span): current close plotted 26 periods back
    # Not used for signals as it requires future data
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Kumo (Cloud) top and bottom
    kumo_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    kumo_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # Volume filter: current volume > 1.8x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, 30)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        if np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or np.isnan(vol_ma_30[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_30[i]
        
        # Volume filter
        volume_ok = vol > 1.8 * vol_ma
        
        # Kumo twist: Senkou A crossing above/below Senkou B
        # We detect twist by comparing current and previous values
        if i > 0:
            senkou_a_prev = senkou_a_aligned[i-1]
            senkou_b_prev = senkou_b_aligned[i-1]
            kumo_top_prev = np.maximum(senkou_a_prev, senkou_b_prev)
            kumo_bottom_prev = np.minimum(senkou_a_prev, senkou_b_prev)
            
            # Bullish twist: Senkou A crosses above Senkou B
            bullish_twist = senkou_a_prev <= senkou_b_prev and senkou_a > senkou_b
            # Bearish twist: Senkou A crosses below Senkou B
            bearish_twist = senkou_a_prev >= senkou_b_prev and senkou_a < senkou_b
        else:
            bullish_twist = False
            bearish_twist = False
        
        # Price above/below cloud
        price_above_kumo = price > kumo_top
        price_below_kumo = price < kumo_bottom
        
        if position == 0:
            # Long: bullish Kumo twist + price above cloud + volume
            if bullish_twist and price_above_kumo and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: bearish Kumo twist + price below cloud + volume
            elif bearish_twist and price_below_kumo and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below Kumo (cloud support broken)
            if price < kumo_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above Kumo (cloud resistance broken)
            if price > kumo_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals