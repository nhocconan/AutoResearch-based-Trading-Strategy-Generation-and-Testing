#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Ichimoku_Cloud_Kijun_Trend"
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
    
    # Daily data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    high_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((high_senkou_b + low_senkou_b) / 2)
    
    # Chikou Span (Lagging Span): close shifted 26 periods behind (not used for signals)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a, additional_delay_bars=26)  # Shifted 26 periods ahead
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b, additional_delay_bars=26)  # Shifted 26 periods ahead
    
    # Kumo (Cloud) top and bottom
    kumo_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    kumo_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # Volume spike: current volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Tenkan crosses above Kijun, price above cloud, volume spike
            tenkan_prev = tenkan_aligned[i-1] if i > 0 else tenkan_aligned[i]
            kijun_prev = kijun_aligned[i-1] if i > 0 else kijun_aligned[i]
            tk_cross_up = (tenkan_prev <= kijun_prev) and (tenkan_aligned[i] > kijun_aligned[i])
            price_above_cloud = close[i] > kumo_top[i]
            
            # Short: Tenkan crosses below Kijun, price below cloud, volume spike
            tk_cross_down = (tenkan_prev >= kijun_prev) and (tenkan_aligned[i] < kijun_aligned[i])
            price_below_cloud = close[i] < kumo_bottom[i]
            
            if tk_cross_up and price_above_cloud and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            elif tk_cross_down and price_below_cloud and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Tenkan crosses below Kijun OR price drops below cloud
            tenkan_prev = tenkan_aligned[i-1] if i > 0 else tenkan_aligned[i]
            kijun_prev = kijun_aligned[i-1] if i > 0 else kijun_aligned[i]
            tk_cross_down = (tenkan_prev >= kijun_prev) and (tenkan_aligned[i] < kijun_aligned[i])
            price_below_cloud = close[i] < kumo_top[i]  # Exit if price drops below cloud top
            
            if tk_cross_down or price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Tenkan crosses above Kijun OR price rises above cloud
            tenkan_prev = tenkan_aligned[i-1] if i > 0 else tenkan_aligned[i]
            kijun_prev = kijun_aligned[i-1] if i > 0 else kijun_aligned[i]
            tk_cross_up = (tenkan_prev <= kijun_prev) and (tenkan_aligned[i] > kijun_aligned[i])
            price_above_cloud = close[i] > kumo_bottom[i]  # Exit if price rises above cloud bottom
            
            if tk_cross_up or price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals