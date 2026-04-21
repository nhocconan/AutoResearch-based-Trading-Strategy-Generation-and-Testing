#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1dTrend
Hypothesis: 6h Ichimoku Tenkan/Kijun cross with 1d cloud filter (price above/below cloud) and volume confirmation. 
The Ichimoku cloud acts as dynamic support/resistance, while the TK cross signals momentum shifts. 
Using 1d timeframe for cloud ensures we only trade in alignment with the higher timeframe trend, 
reducing false signals in choppy markets. Designed for low trade frequency (~15-25/year) to minimize fee drag 
while capturing strong trending moves in both bull and bear markets via cloud position filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Ichimoku cloud)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d Ichimoku Cloud Calculation ===
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    tenkan_sen = (pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max() + 
                  pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    tenkan_sen = tenkan_sen.values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    kijun_sen = (pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max() + 
                 pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    kijun_sen = kijun_sen.values
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    period_senkou_b = 52
    senkou_b = (pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max() + 
                pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2
    senkou_b = senkou_b.values
    
    # Align Ichimoku components to 6h timeframe (with proper look-ahead avoidance)
    # Tenkan and Kijun are aligned with 1-bar delay (completed 1d values)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    # Senkou spans need extra delay because they are plotted ahead
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a, additional_delay_bars=26)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b, additional_delay_bars=26)
    
    # === 6h Volume Filter (2.0x 20-period MA) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) 
            or np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) 
            or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = prices['close'].iloc[i]
        volume_now = volume[i]
        tenkan = tenkan_aligned[i]
        kijun = kijun_aligned[i]
        senkou_a = senkou_a_aligned[i]
        senkou_b = senkou_b_aligned[i]
        vol_avg = vol_ma[i]
        
        # Cloud top and bottom
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        
        # Volume spike: current volume > 2.0x average
        volume_spike = volume_now > 2.0 * vol_avg
        
        if position == 0:
            # Enter long: price above cloud, TK cross bullish (Tenkan > Kijun), volume spike
            long_condition = (price > cloud_top) and (tenkan > kijun) and volume_spike
            # Enter short: price below cloud, TK cross bearish (Tenkan < Kijun), volume spike
            short_condition = (price < cloud_bottom) and (tenkan < kijun) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Exit conditions
            if position == 1:
                # Exit long: price falls below cloud bottom OR TK cross turns bearish
                if (price < cloud_bottom) or (tenkan < kijun):
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price rises above cloud top OR TK cross turns bullish
                if (price > cloud_top) or (tenkan > kijun):
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1dTrend"
timeframe = "6h"
leverage = 1.0