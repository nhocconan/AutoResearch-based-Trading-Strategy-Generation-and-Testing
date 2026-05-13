#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filtered_TK_Cross
Hypothesis: Ichimoku cloud with TK cross signals on 6h timeframe, filtered by daily trend (price above/below daily Kumo cloud) and volume confirmation. 
The Ichimoku system provides multi-factor trend confirmation (Tenkan/Kijun cross, cloud position, future cloud) which reduces whipsaws in sideways markets. 
Daily cloud filter ensures alignment with higher timeframe trend, avoiding counter-trend trades. Volume spike (>2x 24-period average) confirms momentum. 
Designed for 6-12 trades per month (72-144/year) to balance opportunity with fee minimization on 6h timeframe.
"""

name = "6h_Ichimoku_Cloud_Filtered_TK_Cross"
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
    volume = prices['volume'].values
    
    # Get 1d data for cloud filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Ichimoku components for cloud filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    senkou_span_b = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                      pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2)
    
    # Align 1d Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values, additional_delay_bars=26)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values, additional_delay_bars=26)
    
    # 6h Ichimoku components for TK cross signal
    tenkan_sen_6h = (pd.Series(high).rolling(window=9, min_periods=9).max() + 
                     pd.Series(low).rolling(window=9, min_periods=9).min()) / 2
    kijun_sen_6h = (pd.Series(high).rolling(window=26, min_periods=26).max() + 
                    pd.Series(low).rolling(window=26, min_periods=26).min()) / 2
    
    # Volume confirmation: current volume > 2.0x 24-period average (4 days on 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after warmup for Senkou Span B
        # Determine cloud boundaries (using aligned Senkou Spans)
        cloud_top = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        if position == 0:
            # LONG: TK cross bullish + price above cloud + volume filter
            if (tenkan_sen_6h[i] > kijun_sen_6h[i] and  # TK cross bullish
                close[i] > cloud_top and                # Price above cloud
                volume_filter[i]):                      # Volume confirmation
                signals[i] = 0.25
                position = 1
            # SHORT: TK cross bearish + price below cloud + volume filter
            elif (tenkan_sen_6h[i] < kijun_sen_6h[i] and  # TK cross bearish
                  close[i] < cloud_bottom and             # Price below cloud
                  volume_filter[i]):                      # Volume confirmation
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TK cross bearish OR price drops below cloud
            if (tenkan_sen_6h[i] < kijun_sen_6h[i]) or \
               (close[i] < cloud_top):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TK cross bullish OR price rises above cloud
            if (tenkan_sen_6h[i] > kijun_sen_6h[i]) or \
               (close[i] > cloud_bottom):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals