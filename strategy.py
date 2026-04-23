#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud (Tenkan/Kijun) cross with 1d cloud filter and volume confirmation.
Long when Tenkan crosses above Kijun AND price is above 1d cloud (Senkou Span A/B) AND volume > 1.5x 20-period average.
Short when Tenkan crosses below Kijun AND price is below 1d cloud AND volume > 1.5x 20-period average.
Exit on opposite Tenkan/Kijun cross or when price re-enters the 1d cloud.
Uses 1d HTF for cloud (avoids whipsaws in ranging markets). Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Ichimoku cloud (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    tenkan_1d = (pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max() + 
                 pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    kijun_1d = (pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max() + 
                pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2).shift(period_kijun)  # shifted 26 periods ahead
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    senkou_b_1d = ((pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max() + 
                    pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2).shift(period_kijun)
    
    tenkan_1d = tenkan_1d.values
    kijun_1d = kijun_1d.values
    senkou_a_1d = senkou_a_1d.values
    senkou_b_1d = senkou_b_1d.values
    
    # Align HTF Ichimoku to LTF (6h)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Calculate 6h Ichimoku components (for entry signals)
    tenkan_6h = (pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max() + 
                 pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    kijun_6h = (pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max() + 
                pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    tenkan_6h = tenkan_6h.values
    kijun_6h = kijun_6h.values
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(period_kijun - 1, period_senkou_b + period_kijun - 1, 20)  # Ichimoku needs 52+26=78 bars for cloud, 20 for volume
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        tenkan_6h_val = tenkan_6h[i]
        kijun_6h_val = kijun_6h[i]
        tenkan_1d_val = tenkan_1d_aligned[i]
        kijun_1d_val = kijun_1d_aligned[i]
        senkou_a = senkou_a_1d_aligned[i]
        senkou_b = senkou_b_1d_aligned[i]
        vol_ma_val = vol_ma[i]
        
        # Determine 1d cloud boundaries (upper and lower)
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        
        # Calculate Tenkan/Kijun cross for 6h (entry signal)
        if i >= start_idx + 1:
            tenkan_prev = tenkan_6h[i-1]
            kijun_prev = kijun_6h[i-1]
            tk_cross_up = (tenkan_6h_val > kijun_6h_val) and (tenkan_prev <= kijun_prev)
            tk_cross_down = (tenkan_6h_val < kijun_6h_val) and (tenkan_prev >= kijun_prev)
        else:
            tk_cross_up = False
            tk_cross_down = False
        
        if position == 0:
            # Long: Tenkan/Kijun cross up AND price above 1d cloud AND volume spike
            if tk_cross_up and price > cloud_top and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Tenkan/Kijun cross down AND price below 1d cloud AND volume spike
            elif tk_cross_down and price < cloud_bottom and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Tenkan/Kijun cross down OR price re-enters 1d cloud
                if tk_cross_down or price < cloud_top:
                    exit_signal = True
            elif position == -1:
                # Short exit: Tenkan/Kijun cross up OR price re-enters 1d cloud
                if tk_cross_up or price > cloud_bottom:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Ichimoku_TK_Cross_1dCloud_Filter_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0