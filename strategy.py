#!/usr/bin/env python3
name = "6h_Ichimoku_TK_Cross_1dCloud_Filter_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d Ichimoku Cloud (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Tenkan-sen (9-period)
    tenkan_sen_1d = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                     pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (26-period)
    kijun_sen_1d = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                    pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (leading span A)
    senkou_span_a_1d = (tenkan_sen_1d + kijun_sen_1d) / 2
    # Senkou Span B (leading span B)
    senkou_span_b_1d = (pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                        pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2
    
    # Align to 6h timeframe
    tenkan_sen_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen_1d.values)
    kijun_sen_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen_1d.values)
    senkou_span_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d.values)
    senkou_span_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d.values)
    
    # Cloud top and bottom
    cloud_top_1d = np.maximum(senkou_span_a_1d_aligned, senkou_span_b_1d_aligned)
    cloud_bottom_1d = np.minimum(senkou_span_a_1d_aligned, senkou_span_b_1d_aligned)
    
    # 6h Tenkan/Kijun cross
    tenkan_sen_6h = (pd.Series(high).rolling(window=9, min_periods=9).max() + 
                     pd.Series(low).rolling(window=9, min_periods=9).min()) / 2
    kijun_sen_6h = (pd.Series(high).rolling(window=26, min_periods=26).max() + 
                    pd.Series(low).rolling(window=26, min_periods=26).min()) / 2
    tk_cross_up = tenkan_sen_6h > kijun_sen_6h
    tk_cross_down = tenkan_sen_6h < kijun_sen_6h
    
    # Volume confirmation: spike > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 26, 20)  # Wait for Ichimoku and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(tenkan_sen_1d_aligned[i]) or np.isnan(kijun_sen_1d_aligned[i]) or \
           np.isnan(cloud_top_1d[i]) or np.isnan(cloud_bottom_1d[i]) or \
           np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TK cross up + price above cloud + volume spike
            if tk_cross_up[i] and close[i] > cloud_top_1d[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: TK cross down + price below cloud + volume spike
            elif tk_cross_down[i] and close[i] < cloud_bottom_1d[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TK cross down or price below cloud
            if tk_cross_down[i] or close[i] < cloud_bottom_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TK cross up or price above cloud
            if tk_cross_up[i] or close[i] > cloud_top_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Ichimoku TK cross with 1d cloud filter and volume confirmation captures trend changes with institutional validation.
# Long when Tenkan crosses above Kijun (bullish momentum) with price above 1d cloud (bullish trend) and volume spike.
# Short when Tenkan crosses below Kijun (bearish momentum) with price below 1d cloud (bearish trend) and volume spike.
# The 1d cloud acts as a higher-timeframe trend filter, ensuring trades align with the dominant trend.
# Volume spike (>2x average) confirms conviction behind the signal.
# Designed for 6h timeframe to target 12-37 trades/year, avoiding overtrading.
# Works in bull markets (TK cross up in bullish cloud) and bear markets (TK cross down in bearish cloud).