#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_TK_Cross_1dTrend_Filter_v1
Hypothesis: On 6h timeframe, Ichimoku Tenkan-Kijun (TK) cross signals aligned with 1d trend (price vs Kumo) and confirmed by volume produce high-probability trend-following entries. The Ichimoku cloud acts as dynamic support/resistance, while the TK cross provides momentum timing. Volume confirmation reduces false signals. Works in both bull and bear markets by only taking trades in the direction of the 1d trend filter. Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Load 1d data ONCE before loop for HTF trend filter (Ichimoku) and volume MA
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_9 = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_26 = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_52 = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_52 + min_low_52) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Kumo (Cloud) top and bottom
    kumohigh = np.maximum(senkou_a_aligned, senkou_b_aligned)
    kumolow = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # 1d trend filter: price above/below Kumo
    uptrend = close > kumohigh
    downtrend = close < kumolow
    
    # 6h volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 52 for Senkou B, 26 for Kijun, 9 for Tenkan, 20 for volume)
    start_idx = max(52, 26, 9, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or
            np.isnan(kumohigh[i]) or
            np.isnan(kumolow[i]) or
            np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Ichimoku TK cross (Tenkan crossing Kijun)
        tk_cross_up = (tenkan_aligned[i-1] <= kijun_aligned[i-1]) and (tenkan_aligned[i] > kijun_aligned[i])
        tk_cross_down = (tenkan_aligned[i-1] >= kijun_aligned[i-1]) and (tenkan_aligned[i] < kijun_aligned[i])
        
        # Volume confirmation
        volume_spike = volume[i] > 1.8 * vol_ma_20[i]
        
        # Long logic: TK cross up in uptrend with volume, price above cloud
        if tk_cross_up and uptrend[i] and volume_spike and (close[i] > kumohigh[i]):
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: TK cross down in downtrend with volume, price below cloud
        elif tk_cross_down and downtrend[i] and volume_spike and (close[i] < kumolow[i]):
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: TK cross in opposite direction or price enters cloud
        elif position == 1 and (tk_cross_down or close[i] <= kumohigh[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (tk_cross_up or close[i] >= kumolow[i]):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_1dTrend_Filter_v1"
timeframe = "6h"
leverage = 1.0