#!/usr/bin/env python3
"""
6h_Ichimoku_TKCross_1dTrendFilter
Hypothesis: Uses Ichimoku Tenkan/Kijun cross on 6h as entry signal, filtered by 1d EMA50 trend direction. In uptrend, take TK cross bullish; in downtrend, take TK cross bearish. Includes volume confirmation and minimum holding period to reduce whipsaw. Works in both bull and bear markets by aligning with higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    tenkan_sen = np.full(n, np.nan)
    for i in range(period_tenkan - 1, n):
        period_high = high[i - period_tenkan + 1:i + 1].max()
        period_low = low[i - period_tenkan + 1:i + 1].min()
        tenkan_sen[i] = (period_high + period_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    kijun_sen = np.full(n, np.nan)
    for i in range(period_kijun - 1, n):
        period_high = high[i - period_kijun + 1:i + 1].max()
        period_low = low[i - period_kijun + 1:i + 1].min()
        kijun_sen[i] = (period_high + period_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    senkou_span_b = np.full(n, np.nan)
    for i in range(period_senkou_b - 1, n):
        period_high = high[i - period_senkou_b + 1:i + 1].max()
        period_low = low[i - period_senkou_b + 1:i + 1].min()
        senkou_span_b[i] = (period_high + period_low) / 2
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        k = 2 / (50 + 1)
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = close_1d[i] * k + ema_50_1d[i-1] * (1 - k)
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            vol_ma[i] = np.mean(volume[0:i+1])
        else:
            vol_ma[i] = np.mean(volume[i-20+1:i+1])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(period_kijun, period_senkou_b)  # Wait for Kijun and Senkou B
    
    for i in range(start_idx, n):
        # Skip if any required values are NaN
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        bars_since_entry += 1
        
        # Determine trend from 1d EMA50: price above EMA = uptrend, below = downtrend
        is_uptrend = close[i] > ema_50_1d_aligned[i]
        
        # TK cross signals
        tk_cross_bullish = tenkan_sen[i] > kijun_sen[i] and tenkan_sen[i-1] <= kijun_sen[i-1]
        tk_cross_bearish = tenkan_sen[i] < kijun_sen[i] and tenkan_sen[i-1] >= kijun_sen[i-1]
        
        if position == 0:
            # Enter long: TK cross bullish in uptrend with volume confirmation
            if tk_cross_bullish and is_uptrend and vol_spike[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Enter short: TK cross bearish in downtrend with volume confirmation
            elif tk_cross_bearish and not is_uptrend and vol_spike[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Exit conditions: minimum 3 bars hold, then exit on TK cross bearish or trend change
            if bars_since_entry >= 3:
                if tk_cross_bearish or not is_uptrend or not vol_spike[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25  # Hold during minimum period
        
        elif position == -1:
            # Exit conditions: minimum 3 bars hold, then exit on TK cross bullish or trend change
            if bars_since_entry >= 3:
                if tk_cross_bullish or is_uptrend or not vol_spike[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25  # Hold during minimum period
    
    return signals

name = "6h_Ichimoku_TKCross_1dTrendFilter"
timeframe = "6h"
leverage = 1.0