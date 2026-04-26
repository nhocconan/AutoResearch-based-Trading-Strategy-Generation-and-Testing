#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_TK_Cross_1dTrendFilter_VolumeConfirm
Hypothesis: Ichimoku TK cross on 6h with 1d trend filter (price vs Kumo twist) and volume confirmation.
Only trade when TK cross occurs in direction of 1d trend (price above/below Kumo) and volume > 1.5x median.
Designed for both bull and bear markets via 1d trend filter and volume confirmation to avoid false signals.
Target: 50-150 total trades over 4 years = 12-37/year. Size: 0.25.
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
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # need 26*2 for Senkou Span B
        return np.zeros(n)
    
    # Calculate Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max()
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min()
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max()
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min()
    kijun = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    period_senkou_b = 52
    high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max()
    low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()
    senkou_b = ((high_senkou_b + low_senkou_b) / 2)
    
    # Current Kumo (cloud) boundaries: Senkou Span A and B shifted back 26 periods
    # So we use the values that were plotted 26 periods ago
    senkou_a_lagged = senkou_a.shift(26)
    senkou_b_lagged = senkou_b.shift(26)
    
    # Kumo top and bottom
    kumo_top = np.maximum(senkou_a_lagged, senkou_b_lagged)
    kumo_bottom = np.minimum(senkou_a_lagged, senkou_b_lagged)
    
    # Kumo twist: bullish when Senkou A > Senkou B, bearish when Senkou A < Senkou B
    kumo_twist_bullish = senkou_a_lagged > senkou_b_lagged
    kumo_twist_bearish = senkou_a_lagged < senkou_b_lagged
    
    # TK cross: bullish when Tenkan crosses above Kijun, bearish when Tenkan crosses below Kijun
    tk_cross_bullish = (tenkan > kijun) & (tenkan.shift(1) <= kijun.shift(1))
    tk_cross_bearish = (tenkan < kijun) & (tenkan.shift(1) >= kijun.shift(1))
    
    # Align 1d HTF trend filter: price vs Kumo (using 1d Ichimoku)
    # Calculate 1d Ichimoku for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Tenkan-sen (9-period)
    high_tenkan_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max()
    low_tenkan_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min()
    tenkan_1d = (high_tenkan_1d + low_tenkan_1d) / 2
    
    # 1d Kijun-sen (26-period)
    high_kijun_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max()
    low_kijun_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min()
    kijun_1d = (high_kijun_1d + low_kijun_1d) / 2
    
    # 1d Senkou Span A
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    
    # 1d Senkou Span B (52-period)
    high_senkou_b_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max()
    low_senkou_b_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min()
    senkou_b_1d = ((high_senkou_b_1d + low_senkou_b_1d) / 2)
    
    # 1d Kumo (cloud) boundaries shifted back 26 periods
    senkou_a_1d_lagged = senkou_a_1d.shift(26)
    senkou_b_1d_lagged = senkou_b_1d.shift(26)
    kumo_top_1d = np.maximum(senkou_a_1d_lagged, senkou_b_1d_lagged)
    kumo_bottom_1d = np.minimum(senkou_a_1d_lagged, senkou_b_1d_lagged)
    
    # 1d trend filter: price above Kumo = uptrend, price below Kumo = downtrend
    price_above_kumo_1d = close_1d > kumo_top_1d
    price_below_kumo_1d = close_1d < kumo_bottom_1d
    
    # Align 1d HTF arrays to 6h timeframe
    kumo_top_1d_aligned = align_htf_to_ltf(prices, df_1d, kumo_top_1d.values)
    kumo_bottom_1d_aligned = align_htf_to_ltf(prices, df_1d, kumo_bottom_1d.values)
    price_above_kumo_1d_aligned = align_htf_to_ltf(prices, df_1d, price_above_kumo_1d.values)
    price_below_kumo_1d_aligned = align_htf_to_ltf(prices, df_1d, price_below_kumo_1d.values)
    kumo_twist_bullish_1d_aligned = align_htf_to_ltf(prices, df_1d, kumo_twist_bullish.values)
    kumo_twist_bearish_1d_aligned = align_htf_to_ltf(prices, df_1d, kumo_twist_bearish.values)
    
    # Volume spike: volume > 1.5x 20-period median volume
    volume_series = pd.Series(volume)
    vol_median_20 = volume_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (1.5 * vol_median_20)
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 52 for Senkou B, 26 for Senkou A shift, 20 for volume median
    start_idx = 52 + 26  # 78
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or
            np.isnan(tk_cross_bullish[i]) or np.isnan(tk_cross_bearish[i]) or
            np.isnan(kumo_top_1d_aligned[i]) or np.isnan(kumo_bottom_1d_aligned[i]) or
            np.isnan(price_above_kumo_1d_aligned[i]) or np.isnan(price_below_kumo_1d_aligned[i]) or
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        vol_spike = volume_spike[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Long: TK cross bullish + price above 1d Kumo + volume spike
            long_entry = (tk_cross_bullish[i] and 
                         price_above_kumo_1d_aligned[i] and 
                         vol_spike)
            # Short: TK cross bearish + price below 1d Kumo + volume spike
            short_entry = (tk_cross_bearish[i] and 
                          price_below_kumo_1d_aligned[i] and 
                          vol_spike)
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when TK cross bearish or price falls below Kumo
            if tk_cross_bearish[i] or (close_val < kumo_bottom[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when TK cross bullish or price rises above Kumo
            if tk_cross_bullish[i] or (close_val > kumo_top[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_1dTrendFilter_VolumeConfirm"
timeframe = "6h"
leverage = 1.0