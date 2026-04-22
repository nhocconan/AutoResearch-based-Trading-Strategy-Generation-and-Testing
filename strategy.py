#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku Tenkan-sen (Conversion Line): (9-period high + low) / 2
    high_9 = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Ichimoku Kijun-sen (Base Line): (26-period high + low) / 2
    high_26 = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Ichimoku Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Ichimoku Senkou Span B (Leading Span B): (52-period high + low) / 2
    high_52 = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Volume spike filter (20-period average on 6h)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any data is not ready
        if (np.isnan(tenkan_6h[i]) or 
            np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or 
            np.isnan(senkou_b_6h[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        tenkan_val = tenkan_6h[i]
        kijun_val = kijun_6h[i]
        senkou_a_val = senkou_a_6h[i]
        senkou_b_val = senkou_b_6h[i]
        
        # Cloud top and bottom
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        # Volume filter: current volume > 2.0 * 20-day average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: TK cross bullish + price above cloud + volume spike
            if (tenkan_val > kijun_val and 
                price > cloud_top and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short conditions: TK cross bearish + price below cloud + volume spike
            elif (tenkan_val < kijun_val and 
                  price < cloud_bottom and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: TK cross reverses or price re-enters cloud
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when TK cross turns bearish or price drops below cloud
                if tenkan_val < kijun_val or price < cloud_bottom:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when TK cross turns bullish or price rises above cloud
                if tenkan_val > kijun_val or price > cloud_top:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_Cloud_Filter_Volume"
timeframe = "6h"
leverage = 1.0