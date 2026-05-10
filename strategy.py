#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Rotation_1dTrend_Filter
Hypothesis: Use Ichimoku cloud from 1d timeframe for trend direction (price above/below cloud) and TK cross from 6d for entry timing.
The Ichimoku cloud acts as dynamic support/resistance that adapts to volatility, working in both bull and bear markets.
In bull markets, price tends to stay above cloud; in bear markets, below cloud.
TK cross provides timely entries within the trend, reducing whipsaw.
Combined with volume confirmation to avoid low-probability breakouts.
Target: 15-30 trades/year, avoiding overtrading while capturing sustained moves.
"""

name = "6h_Ichimoku_Cloud_Rotation_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for Ichimoku cloud (trend filter)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 52:  # Need at least 52 periods for Senkou B
        return np.zeros(n)
    
    # Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    tenkan_sen = (pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max() + 
                  pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    kijun_sen = (pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max() + 
                 pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    senkou_span_b = ((pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max() + 
                      pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2)
    
    # Align Ichimoku components to 6h timeframe (wait for 1d bar to close)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values, additional_delay_bars=1)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values, additional_delay_bars=1)
    
    # Get 6d price and volume for entry signals
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # TK Cross: Tenkan-sen crosses above/below Kijun-sen
    tk_cross_up = (tenkan_sen_aligned > kijun_sen_aligned) & (tenkan_sen_aligned.shift(1) <= kijun_sen_aligned.shift(1))
    tk_cross_down = (tenkan_sen_aligned < kijun_sen_aligned) & (tenkan_sen_aligned.shift(1) >= kijun_sen_aligned.shift(1))
    
    # Cloud: price above/below cloud
    # Cloud top = max(Senkou A, Senkou B), cloud bottom = min(Senkou A, Senkou B)
    cloud_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    cloud_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    # Volume filter: current volume > 1.3x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Ichimoku (52 periods) + TK cross needs prior values
    start_idx = 53
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or
            np.isnan(cloud_top[i]) or
            np.isnan(cloud_bottom[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above cloud AND TK cross up with volume
            if price_above_cloud[i] and tk_cross_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below cloud AND TK cross down with volume
            elif price_below_cloud[i] and tk_cross_down[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below cloud OR TK cross down (momentum loss)
            if close[i] < cloud_top[i] or tk_cross_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above cloud OR TK cross up (momentum loss)
            if close[i] > cloud_bottom[i] or tk_cross_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals