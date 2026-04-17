#!/usr/bin/env python3
"""
6h Ichimoku Cloud + 1d Tenkan-Kijun Cross + Volume Spike
Long: Tenkan > Kijun + Price > Kumo + Volume > 2x 6h MA
Short: Tenkan < Kijun + Price < Kumo + Volume > 2x 6h MA
Exit: Opposite TK cross or price crosses Kumo
Target: 15-30 trades/year per symbol, uses Ichimoku for trend + momentum, volume for confirmation
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
    
    # 1d Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = low = df_1d['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                 pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2)
    # Chikou Span (Lagging Span): Close shifted 26 periods back (not used for signals)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_1d = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_1d = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_a_1d = align_htf_to_ltf(prices, df_1d, senkou_a.values, additional_delay_bars=26)
    senkou_b_1d = align_htf_to_ltf(prices, df_1d, senkou_b.values, additional_delay_bars=26)
    
    # 6h volume spike filter (2x 20-period MA)
    df_6h = get_htf_data(prices, '6h')
    volume_ma_20 = pd.Series(df_6h['volume']).rolling(window=20, min_periods=20).mean()
    volume_ma_20_6h = align_htf_to_ltf(prices, df_6h, volume_ma_20.values)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_sen_1d[i]) or np.isnan(kijun_sen_1d[i]) or 
            np.isnan(senkou_a_1d[i]) or np.isnan(senkou_b_1d[i]) or 
            np.isnan(volume_ma_20_6h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20_6h[i]
        tenkan = tenkan_sen_1d[i]
        kijun = kijun_sen_1d[i]
        span_a = senkou_a_1d[i]
        span_b = senkou_b_1d[i]
        
        # Kumo (Cloud) top and bottom
        kumo_top = max(span_a, span_b)
        kumo_bottom = min(span_a, span_b)
        
        if position == 0:
            # Long: TK bullish + Price above Kumo + Volume spike
            if tenkan > kijun and price > kumo_top and vol > 2.0 * vol_ma:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: TK bearish + Price below Kumo + Volume spike
            elif tenkan < kijun and price < kumo_bottom and vol > 2.0 * vol_ma:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: TK bearish or price drops below Kumo
            if tenkan < kijun or price < kumo_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TK bullish or price rises above Kumo
            if tenkan > kijun or price > kumo_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_Volume"
timeframe = "6h"
leverage = 1.0