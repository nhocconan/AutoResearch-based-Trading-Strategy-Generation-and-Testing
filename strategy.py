#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_TK_Cross_1dTrendFilter
Hypothesis: Ichimoku TK cross on 6h with 1d cloud filter and volume confirmation.
Long when TK crosses above AND price > 1d cloud (bullish regime). Short when TK crosses below AND price < 1d cloud (bearish regime).
Volume spike filter reduces false signals. Designed for both bull and bear markets via 1d cloud regime filter.
Targets 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25 to minimize fee drag.
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
    
    # Load 1d data ONCE before loop for HTF cloud filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 52 for Ichimoku
        return np.zeros(n)
    
    # Calculate 1d Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                      pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2).shift(26)
    
    # Align Ichimoku components to 6h timeframe (wait for 1d bar to close)
    tenkan_sen_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_6h = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_6h = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_6h = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    # Calculate 6h TK cross (Tenkan-sen crosses Kijun-sen)
    tk_cross_above = (tenkan_sen_6h > kijun_sen_6h) & (np.roll(tenkan_sen_6h, 1) <= np.roll(kijun_sen_6h, 1))
    tk_cross_below = (tenkan_sen_6h < kijun_sen_6h) & (np.roll(tenkan_sen_6h, 1) >= np.roll(kijun_sen_6h, 1))
    
    # Determine cloud (Kumo) - green when Senkou Span A > Senkou Span B, red when opposite
    cloud_green = senkou_span_a_6h > senkou_span_b_6h  # Bullish cloud
    cloud_red = senkou_span_a_6h < senkou_span_b_6h    # Bearish cloud
    
    # Price relative to cloud
    price_above_cloud = close > np.maximum(senkou_span_a_6h, senkou_span_b_6h)
    price_below_cloud = close < np.minimum(senkou_span_a_6h, senkou_span_b_6h)
    
    # Volume spike: volume > 1.5x 20-period median volume
    volume_series = pd.Series(volume)
    vol_median_20 = volume_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (1.5 * vol_median_20)
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 52 for Ichimoku, 20 for volume median
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen_6h[i]) or
            np.isnan(kijun_sen_6h[i]) or
            np.isnan(senkou_span_a_6h[i]) or
            np.isnan(senkou_span_b_6h[i]) or
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        vol_spike = volume_spike[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Long: TK cross above AND price above cloud (bullish regime) AND volume spike
            long_entry = tk_cross_above[i] and price_above_cloud[i] and vol_spike
            # Short: TK cross below AND price below cloud (bearish regime) AND volume spike
            short_entry = tk_cross_below[i] and price_below_cloud[i] and vol_spike
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when TK cross below OR price drops below cloud
            if tk_cross_below[i] or not price_above_cloud[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when TK cross above OR price rises above cloud
            if tk_cross_above[i] or not price_below_cloud[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_1dTrendFilter"
timeframe = "6h"
leverage = 1.0