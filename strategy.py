#!/usr/bin/env python3
name = "6h_Ichimoku_TK_Cross_Cloud_Filter_1dTrend"
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
    
    # Daily Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    tenkan_sen_1d = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                     pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    tenkan_sen_1d = tenkan_sen_1d.values
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    kijun_sen_1d = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                    pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    kijun_sen_1d = kijun_sen_1d.values
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2, shifted 26 periods ahead
    senkou_span_a_1d = ((tenkan_sen_1d + kijun_sen_1d) / 2)
    # Senkou Span B (Leading Span B): (52-period high + low)/2, shifted 26 periods ahead
    senkou_span_b_1d = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                         pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2)
    senkou_span_b_1d = senkou_span_b_1d.values
    
    # Align Ichimoku components to 6h (no additional delay needed as Ichimoku uses current bar data)
    tenkan_sen_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen_1d)
    kijun_sen_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen_1d)
    senkou_span_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d)
    senkou_span_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d)
    
    # Kumo (Cloud) boundaries: Senkou Span A and B
    # Cloud top is max of Senkou Span A/B, cloud bottom is min
    # However, since Senkou spans are plotted 26 periods ahead, we need to align them properly
    # For simplicity, we use the current cloud (already shifted in calculation)
    # Cloud top: max(Senkou Span A, Senkou Span B)
    # Cloud bottom: min(Senkou Span A, Senkou Span B)
    # But note: Ichimoku cloud is plotted ahead, so current cloud represents future support/resistance
    # We'll use the current values as is (they represent the cloud for current period)
    cloud_top_1d = np.maximum(senkou_span_a_1d_aligned, senkou_span_b_1d_aligned)
    cloud_bottom_1d = np.minimum(senkou_span_a_1d_aligned, senkou_span_b_1d_aligned)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Session filter: active during major sessions (00-08, 08-16 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # need enough data for Ichimoku calculations
    
    for i in range(start_idx, n):
        # Skip if Ichimoku or volume data not ready
        if (np.isnan(tenkan_sen_1d_aligned[i]) or np.isnan(kijun_sen_1d_aligned[i]) or 
            np.isnan(cloud_top_1d[i]) or np.isnan(cloud_bottom_1d[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Session filter: active during London/NY overlap (08-16 UTC) and Asia (00-08 UTC)
        hour = hours[i]
        in_session = ((0 <= hour <= 8) or (8 <= hour <= 16))
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long conditions: TK cross bullish (Tenkan > Kijun) AND price above cloud
            if (tenkan_sen_1d_aligned[i] > kijun_sen_1d_aligned[i] and
                close[i] > cloud_top_1d[i] and
                volume[i] > vol_ma_20_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: TK cross bearish (Tenkan < Kijun) AND price below cloud
            elif (tenkan_sen_1d_aligned[i] < kijun_sen_1d_aligned[i] and
                  close[i] < cloud_bottom_1d[i] and
                  volume[i] > vol_ma_20_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when TK cross turns bearish OR price drops below cloud
            if (tenkan_sen_1d_aligned[i] < kijun_sen_1d_aligned[i] or
                close[i] < cloud_bottom_1d[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when TK cross turns bullish OR price rises above cloud
            if (tenkan_sen_1d_aligned[i] > kijun_sen_1d_aligned[i] or
                close[i] > cloud_top_1d[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals