#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1dTrend
Hypothesis: Ichimoku Tenkan/Kijun cross with cloud filter and 1-day trend alignment.
Works in bull/bear by using 1d trend to filter direction, reducing false signals in chop.
Designed for low trade frequency (15-30/year) on 6h to minimize fee drag.
"""

name = "6h_Ichimoku_TK_Cross_CloudFilter_1dTrend"
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
    
    # === Get daily data for Ichimoku and trend ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Ichimoku components (9, 26, 52)
    high_9 = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    high_26 = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    high_52 = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    senkou_a = ((tenkan + kijun) / 2)
    senkou_b = ((high_52 + low_52) / 2)
    
    # Align Ichimoku to 6h
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # 1-day EMA50 trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Choppiness filter (avoid choppy markets)
    atr_14 = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values
    highest_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10((highest_14 - lowest_14) / atr_14 / np.log2(14))  # Simplified chop
    
    # Signal parameters
    position_size = 0.25
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Warmup for Ichimoku
    
    for i in range(start_idx, n):
        # Skip if any data invalid
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or
            np.isnan(ema50_1d_6h[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Cloud top/bottom
        cloud_top = np.maximum(senkou_a_6h[i], senkou_b_6h[i])
        cloud_bottom = np.minimum(senkou_a_6h[i], senkou_b_6h[i])
        
        if position == 0:
            # Long: TK cross bullish, price above cloud, uptrend, not choppy
            if (tenkan_6h[i] > kijun_6h[i] and 
                close[i] > cloud_top and 
                close[i] > ema50_1d_6h[i] and
                chop[i] < 61.8):
                signals[i] = position_size
                position = 1
            # Short: TK cross bearish, price below cloud, downtrend, not choppy
            elif (tenkan_6h[i] < kijun_6h[i] and 
                  close[i] < cloud_bottom and 
                  close[i] < ema50_1d_6h[i] and
                  chop[i] < 61.8):
                signals[i] = -position_size
                position = -1
        else:
            # Exit: TK cross reverses or price enters cloud
            if position == 1:
                if (tenkan_6h[i] < kijun_6h[i] or close[i] < cloud_top):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if (tenkan_6h[i] > kijun_6h[i] or close[i] > cloud_bottom):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals