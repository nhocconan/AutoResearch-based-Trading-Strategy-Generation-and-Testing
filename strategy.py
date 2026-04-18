#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1d
Hypothesis: Use Ichimoku on 1d for trend (TK cross + cloud filter) and enter on 6h when price aligns with daily trend. In bull markets, daily TK cross bullish + price above cloud = long bias; in bear markets, daily TK cross bearish + price below cloud = short bias. Enter on 6h pullbacks to the 21-period EMA in the direction of the daily trend. This avoids chasing extended moves and works in both regimes by following the higher timeframe trend. Targets 15-25 trades/year via strict daily alignment requirement.
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
    
    # Get 1d data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters
    tenkan = 9
    kijun = 26
    senkou = 52
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = np.full_like(close_1d, np.nan)
    if len(high_1d) >= tenkan:
        for i in range(tenkan - 1, len(high_1d)):
            tenkan_sen[i] = (np.max(high_1d[i - tenkan + 1:i + 1]) + np.min(low_1d[i - tenkan + 1:i + 1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = np.full_like(close_1d, np.nan)
    if len(high_1d) >= kijun:
        for i in range(kijun - 1, len(high_1d)):
            kijun_sen[i] = (np.max(high_1d[i - kijun + 1:i + 1]) + np.min(low_1d[i - kijun + 1:i + 1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a = np.full_like(close_1d, np.nan)
    if len(tenkan_sen) >= kijun and len(kijun_sen) >= kijun:
        for i in range(kijun - 1, len(close_1d)):
            if not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
                senkou_span_a[i + kijun] = (tenkan_sen[i] + kijun_sen[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b = np.full_like(close_1d, np.nan)
    if len(high_1d) >= senkou:
        for i in range(senkou - 1, len(high_1d)):
            senkou_span_b[i + kijun] = (np.max(high_1d[i - senkou + 1:i + 1]) + np.min(low_1d[i - senkou + 1:i + 1])) / 2
    
    # Align Ichimoku components to 6h
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # 6h EMA(21) for pullback entries
    ema_period = 21
    ema = np.full_like(close, np.nan)
    if len(close) >= ema_period:
        multiplier = 2 / (ema_period + 1)
        ema[ema_period - 1] = np.mean(close[:ema_period])
        for i in range(ema_period, len(close)):
            ema[i] = (close[i] - ema[i-1]) * multiplier + ema[i-1]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(tenkan, kijun, senkou, ema_period) + kijun  # Ensure all data available
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(ema[i])):
            signals[i] = 0.0
            continue
        
        # Determine daily trend bias
        tk_bullish = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        price_above_cloud = close[i] > senkou_span_a_aligned[i] and close[i] > senkou_span_b_aligned[i]
        price_below_cloud = close[i] < senkou_span_a_aligned[i] and close[i] < senkou_span_b_aligned[i]
        
        long_bias = tk_bullish and price_above_cloud
        short_bias = not tk_bullish and price_below_cloud
        
        if position == 0:
            # Long: daily bullish bias + price pulls back to EMA(21) from above
            if long_bias and close[i] <= ema[i] and i > 0 and close[i-1] > ema[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: daily bearish bias + price pulls back to EMA(21) from below
            elif short_bias and close[i] >= ema[i] and i > 0 and close[i-1] < ema[i-1]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: daily bias turns bearish or price breaks below cloud
            if not long_bias or (close[i] < senkou_span_a_aligned[i] and close[i] < senkou_span_b_aligned[i]):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: daily bias turns bullish or price breaks above cloud
            if not short_bias or (close[i] > senkou_span_a_aligned[i] and close[i] > senkou_span_b_aligned[i]):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1d"
timeframe = "6h"
leverage = 1.0