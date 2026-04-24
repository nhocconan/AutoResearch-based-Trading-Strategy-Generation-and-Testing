#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud + 1d TK Cross with Volume Confirmation.
- Uses Ichimoku components: Tenkan-sen (9), Kijun-sen (26), Senkou Span A/B (26, 52 displacement)
- Long when price > cloud AND TK cross bullish (Tenkan > Kijun) with volume spike
- Short when price < cloud AND TK cross bearish (Tenkan < Kijun) with volume spike
- Cloud acts as dynamic support/resistance, TK cross provides momentum signal
- Works in bull/bear via cloud filter (avoid trading against cloud direction)
- Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
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
    
    # Get 1d data ONCE before loop for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_senkou_b + min_low_senkou_b) / 2)
    
    # Align Ichimoku components to 6h timeframe (with 26-bar displacement handled by align_htf_to_ltf)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Volume confirmation: > 2.0x 20-period average volume (stricter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(52, 26, 20) + 26  # +26 for Senkou displacement
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud boundaries (Senkou Span A/B)
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # TK cross momentum (change from previous bar)
        if i > 0:
            tk_momentum = (tenkan_aligned[i] - kijun_aligned[i]) - (tenkan_aligned[i-1] - kijun_aligned[i-1])
        else:
            tk_momentum = 0
        
        if position == 0:
            # Long: Price above cloud AND TK cross bullish AND strengthening with volume spike
            if (close[i] > cloud_top and 
                tenkan_aligned[i] > kijun_aligned[i] and  # Bullish TK cross
                tk_momentum > 0 and  # TK cross strengthening
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price below cloud AND TK cross bearish AND strengthening with volume spike
            elif (close[i] < cloud_bottom and 
                  tenkan_aligned[i] < kijun_aligned[i] and  # Bearish TK cross
                  tk_momentum < 0 and  # TK cross strengthening (more negative)
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price falls below cloud OR TK cross turns bearish
            if close[i] < cloud_top or tenkan_aligned[i] <= kijun_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price rises above cloud OR TK cross turns bullish
            if close[i] > cloud_bottom or tenkan_aligned[i] >= kijun_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TKCross_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0