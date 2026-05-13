#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filtered_TK_Cross_v2
Hypothesis: Ichimoku TK cross (Tenkan/Kijun) on 6h timeframe, filtered by 1d cloud color and volume spike, captures high-probability trend continuations. Reduces false signals by requiring alignment with higher timeframe Ichimoku trend. Works in both bull and bear markets by filtering direction via cloud (price above cloud = bullish bias, below = bearish bias). Targets 15-25 trades/year to minimize fee drag.
"""

name = "6h_Ichimoku_Cloud_Filtered_TK_Cross_v2"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    high_tenkan = pd.Series(df_1d['high']).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(df_1d['low']).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    high_kijun = pd.Series(df_1d['high']).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(df_1d['low']).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 plotted 26 periods ahead
    period_senkou_b = 52
    high_senkou_b = pd.Series(df_1d['high']).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(df_1d['low']).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((high_senkou_b + low_senkou_b) / 2.0)
    
    # Align Ichimoku components to 6h chart
    # Tenkan and Kijun need no additional delay (based on current 1d bar)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    # Senkou spans need 26-period delay because they are plotted ahead
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a, additional_delay_bars=26)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b, additional_delay_bars=26)
    
    # Determine cloud color and boundaries
    # Green cloud (bullish): Senkou A > Senkou B
    # Red cloud (bearish): Senkou A < Senkou B
    cloud_green = senkou_a_aligned > senkou_b_aligned
    cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # Volume confirmation: current volume > 2.0x 24-period average (4 days on 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        if position == 0:
            # LONG: TK cross bullish (Tenkan crosses above Kijun) + price above cloud + volume
            tk_cross_bullish = (tenkan_aligned[i] > kijun_aligned[i]) and (tenkan_aligned[i-1] <= kijun_aligned[i-1])
            price_above_cloud = close[i] > cloud_top[i]
            
            if tk_cross_bullish and price_above_cloud and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: TK cross bearish (Tenkan crosses below Kijun) + price below cloud + volume
            elif (tenkan_aligned[i] < kijun_aligned[i]) and (tenkan_aligned[i-1] >= kijun_aligned[i-1]):
                tk_cross_bearish = True
                price_below_cloud = close[i] < cloud_bottom[i]
                if tk_cross_bearish and price_below_cloud and volume_filter[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TK cross bearish or price drops below cloud
            tk_cross_bearish = (tenkan_aligned[i] < kijun_aligned[i]) and (tenkan_aligned[i-1] >= kijun_aligned[i-1])
            price_below_cloud = close[i] < cloud_top[i]  # Exit if drops below cloud top
            
            if tk_cross_bearish or price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TK cross bullish or price rises above cloud
            tk_cross_bullish = (tenkan_aligned[i] > kijun_aligned[i]) and (tenkan_aligned[i-1] <= kijun_aligned[i-1])
            price_above_cloud = close[i] > cloud_bottom[i]  # Exit if rises above cloud bottom
            
            if tk_cross_bullish or price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals