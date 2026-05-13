#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Twist_1dTrend
Hypothesis: Ichimoku Tenkan/Kijun cross (TK) with cloud filter on 1d timeframe provides 
high-probability trend signals. When price is above/below cloud and TK cross aligns with 
1d trend (close > EMA50), we enter with 0.30 position. Exit when TK cross reverses or 
price re-enters cloud. Designed for 6h timeframe to capture multi-day trends with 
minimal whipsaw in both bull and bear markets.
"""

name = "6h_Ichimoku_Cloud_Twist_1dTrend"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1d data for Ichimoku and trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max() + 
                  pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min()) / 2
    tenkan_sen = tenkan_sen.values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max() + 
                 pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min()) / 2
    kijun_sen = kijun_sen.values
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b = ((pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max() + 
                      pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min()) / 2)
    senkou_span_b = senkou_span_b.values
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # 1d trend filter: EMA(50) on close
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after warmup for Ichimoku
        # Cloud top and bottom (Senkou Span A and B)
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # TK cross signals
        tk_cross = tenkan_sen_aligned[i] - kijun_sen_aligned[i]
        tk_cross_prev = tenkan_sen_aligned[i-1] - kijun_sen_aligned[i-1]
        
        if position == 0:
            # LONG: Price above cloud + TK cross bullish + 1d uptrend
            if (close[i] > cloud_top and 
                tk_cross > 0 and tk_cross_prev <= 0 and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.30
                position = 1
            # SHORT: Price below cloud + TK cross bearish + 1d downtrend
            elif (close[i] < cloud_bottom and 
                  tk_cross < 0 and tk_cross_prev >= 0 and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TK cross turns bearish OR price re-enters cloud
            if (tk_cross < 0) or (close[i] < cloud_top):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: TK cross turns bullish OR price re-enters cloud
            if (tk_cross > 0) or (close[i] > cloud_bottom):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals