#!/usr/bin/env python3
name = "12h_ICHIMOKU_KUMO_BREAKOUT_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku Cloud (primary trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Ichimoku Cloud components (standard settings)
    high_9 = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (high_9 + low_9) / 2
    
    high_26 = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun_sen = (high_26 + low_26) / 2
    
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    senkou_span_b = (pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values + 
                     pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values) / 2
    
    # Align Ichimoku components to 12h timeframe
    tenkan_sen_12h = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_12h = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_12h = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_12h = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Kumo (cloud) boundaries
    kumo_top = np.maximum(senkou_span_a_12h, senkou_span_b_12h)
    kumo_bottom = np.minimum(senkou_span_a_12h, senkou_span_b_12h)
    
    # Get 12h data for volume filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(df_12h['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma)
    volume_ok = df_12h['volume'].values > (vol_ma_aligned * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Wait for Ichimoku calculation
    
    for i in range(start_idx, n):
        if np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or np.isnan(vol_ma_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Kumo (cloud) + above Kijun + volume spike
            if close[i] > kumo_top[i] and close[i] > kijun_sen_12h[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Kumo (cloud) + below Kijun + volume spike
            elif close[i] < kumo_bottom[i] and close[i] < kijun_sen_12h[i] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to opposite side of Kumo or breaks in opposite direction
            if position == 1:
                if close[i] < kumo_bottom[i] or close[i] < kijun_sen_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > kumo_top[i] or close[i] > kijun_sen_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals