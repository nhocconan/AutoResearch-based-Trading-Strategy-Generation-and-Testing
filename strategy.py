#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Trend_1D_Volume_Confirmation
Hypothesis: On 6h timeframe, use Ichimoku cloud for trend direction with TK cross for entry timing, filtered by 1d volume surge. 
Ichimoku provides robust trend identification (cloud) and momentum signals (TK cross), while 1d volume filter ensures trades occur during high conviction periods. 
Works in bull/bear markets as cloud acts as dynamic support/resistance and TK cross captures momentum shifts. Targets 15-25 trades/year to minimize fee drag.
"""

name = "6h_Ichimoku_Cloud_Trend_1D_Volume_Confirmation"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for Ichimoku and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    high_52 = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((high_52 + low_52) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou_span = close_1d  # Will be aligned differently
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a, additional_delay_bars=26)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b, additional_delay_bars=26)
    # For Chikou span, we align with 26-period delay to represent lagging nature
    chikou_span_aligned = align_htf_to_ltf(prices, df_1d, chikou_span, additional_delay_bars=26)
    
    # 1d volume filter
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # 6h data for price
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Ichimoku (52) and volume MA (20)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or
            np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(chikou_span_aligned[i]) or
            np.isnan(vol_ma20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below cloud
        # Cloud top is the higher of Senkou Span A and B
        # Cloud bottom is the lower of Senkou Span A and B
        cloud_top = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # TK Cross signals
        tk_cross_bull = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        tk_cross_bear = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
        
        # Volume filter: current 6h volume > 1.5x 1d 20-period MA
        volume_filter = volume[i] > vol_ma20_1d_aligned[i] * 1.5
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Long: price above cloud + TK cross bull + volume
            if price_above_cloud and tk_cross_bull and volume_filter and in_session:
                signals[i] = 0.25
                position = 1
            # Short: price below cloud + TK cross bear + volume
            elif price_below_cloud and tk_cross_bear and volume_filter and in_session:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below cloud or TK cross turns bear
            if not price_above_cloud or not tk_cross_bull:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above cloud or TK cross turns bull
            if not price_below_cloud or not tk_cross_bear:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals