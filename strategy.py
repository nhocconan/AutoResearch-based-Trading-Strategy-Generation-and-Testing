#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filter_12hTrend_VolumeConfirmation
Hypothesis: 6-hour Ichimoku system with 12-hour trend filter (TK cross) and volume confirmation.
Enters long when price breaks above Kumo cloud with bullish TK cross on 12h and volume spike.
Enters short when price breaks below Kumo cloud with bearish TK cross on 12h and volume spike.
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Target: 50-150 total trades over 4 years.
Designed for BTC/ETH - works in bull markets via breakouts, bear markets via trend-following shorts.
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
    
    # Calculate Ichimoku components on 6h timeframe
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (high_senkou_b + low_senkou_b) / 2.0
    
    # Kumo cloud boundaries (shifted forward by 26 periods)
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Kumo top and bottom
    kumo_top = np.maximum(senkou_a_shifted, senkou_b_shifted)
    kumo_bottom = np.minimum(senkou_a_shifted, senkou_b_shifted)
    
    # TK Cross (Tenkan/Kijun cross) on 12h timeframe for trend filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Tenkan and Kijun on 12h
    high_tenkan_12h = pd.Series(high_12h).rolling(window=9, min_periods=9).max().values
    low_tenkan_12h = pd.Series(low_12h).rolling(window=9, min_periods=9).min().values
    tenkan_12h = (high_tenkan_12h + low_tenkan_12h) / 2.0
    
    high_kijun_12h = pd.Series(high_12h).rolling(window=26, min_periods=26).max().values
    low_kijun_12h = pd.Series(low_12h).rolling(window=26, min_periods=26).min().values
    kijun_12h = (high_kijun_12h + low_kijun_12h) / 2.0
    
    tk_cross_bullish = tenkan_12h > kijun_12h
    tk_cross_bearish = tenkan_12h < kijun_12h
    
    # Align TK cross signals to 6h timeframe
    tk_bullish_aligned = align_htf_to_ltf(prices, df_12h, tk_cross_bullish.astype(float))
    tk_bearish_aligned = align_htf_to_ltf(prices, df_12h, tk_cross_bearish.astype(float))
    
    # Volume confirmation: volume > 1.8 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.8 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 52-period Senkou B + 26 shift)
    start_idx = 52 + 26
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or 
            np.isnan(tk_bullish_aligned[i]) or np.isnan(tk_bearish_aligned[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: price above Kumo + bullish TK cross on 12h + volume spike
        if close[i] > kumo_top[i] and tk_bullish_aligned[i] > 0.5 and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: price below Kumo + bearish TK cross on 12h + volume spike
        elif close[i] < kumo_bottom[i] and tk_bearish_aligned[i] > 0.5 and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: price re-enters Kumo cloud
        elif position == 1 and close[i] < kumo_top[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > kumo_bottom[i]:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_Ichimoku_Cloud_Filter_12hTrend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0