#!/usr/bin/env python3
# 6h_Ichimoku_Cloud_Filter_1dTrend_Volume
# Hypothesis: Ichimoku cloud provides dynamic support/resistance. Use TK cross as entry signal,
# filtered by 1d trend direction (via EMA) and volume confirmation.
# Cloud acts as filter: only take longs when price above cloud in uptrend,
# shorts when price below cloud in downtrend.
# This avoids whipsaws in ranging markets and captures trends with momentum confirmation.
# Target: 20-40 trades/year to stay within limits.

name = "6h_Ichimoku_Cloud_Filter_1dTrend_Volume"
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
    
    # Ichimoku components (9, 26, 52)
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    tenkan_sen = (pd.Series(high).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + low)/2
    kijun_sen = (pd.Series(high).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    senkou_b = ((pd.Series(high).rolling(window=52, min_periods=52).max() + 
                 pd.Series(low).rolling(window=52, min_periods=52).min()) / 2).shift(26)
    
    # Chikou Span (Lagging Span): close shifted 26 periods back (not used for signals)
    
    tenkan = tenkan_sen.values
    kijun = kijun_sen.values
    span_a = senkou_a.values
    span_b = senkou_b.values
    
    # Cloud top and bottom
    cloud_top = np.maximum(span_a, span_b)
    cloud_bottom = np.minimum(span_a, span_b)
    
    # TK Cross signals
    tk_cross_up = (tenkan > kijun) & (np.roll(tenkan, 1) <= np.roll(kijun, 1))
    tk_cross_down = (tenkan < kijun) & (np.roll(tenkan, 1) >= np.roll(kijun, 1))
    
    # 1d trend filter: EMA(34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 80  # Ensure sufficient warmup for Ichimoku (52+26)
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(cloud_top[i]) or 
            np.isnan(cloud_bottom[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TK cross up, price above cloud, 1d trend up, volume confirmation
            if (tk_cross_up[i] and 
                close[i] > cloud_top[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: TK cross down, price below cloud, 1d trend down, volume confirmation
            elif (tk_cross_down[i] and 
                  close[i] < cloud_bottom[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: TK cross down OR price drops below cloud OR trend changes
            if (tk_cross_down[i] or 
                close[i] < cloud_bottom[i] or 
                close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: TK cross up OR price rises above cloud OR trend changes
            if (tk_cross_up[i] or 
                close[i] > cloud_top[i] or 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals