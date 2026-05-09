#!/usr/bin/env python3
# 1d_Ichimoku_Kumo_Breakout_1wTrend_Filter
# Hypothesis: Buy when price breaks above Kumo (cloud) with bullish weekly Tenkan/Kijun cross and price > weekly Senkou Span A;
# Sell when price breaks below Kumo with bearish weekly Tenkan/Kijun cross and price < weekly Senkou Span B.
# Uses Ichimoku cloud as dynamic support/resistance and weekly trend filter to avoid counter-trend trades.
# Designed for 10-25 trades/year on 1d timeframe with low turnover to minimize fee drag.

name = "1d_Ichimoku_Kumo_Breakout_1wTrend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 26:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Ichimoku components on daily data
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    high_tenkan = np.full_like(high, np.nan)
    low_tenkan = np.full_like(low, np.nan)
    if len(high) >= period_tenkan:
        for i in range(period_tenkan - 1, len(high)):
            high_tenkan[i] = np.max(high[i - period_tenkan + 1:i + 1])
            low_tenkan[i] = np.min(low[i - period_tenkan + 1:i + 1])
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    high_kijun = np.full_like(high, np.nan)
    low_kijun = np.full_like(low, np.nan)
    if len(high) >= period_kijun:
        for i in range(period_kijun - 1, len(high)):
            high_kijun[i] = np.max(high[i - period_kijun + 1:i + 1])
            low_kijun[i] = np.min(low[i - period_kijun + 1:i + 1])
    kijun = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2 plotted 26 periods ahead
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2 plotted 26 periods ahead
    period_senkou_b = 52
    high_senkou_b = np.full_like(high, np.nan)
    low_senkou_b = np.full_like(low, np.nan)
    if len(high) >= period_senkou_b:
        for i in range(period_senkou_b - 1, len(high)):
            high_senkou_b[i] = np.max(high[i - period_senkou_b + 1:i + 1])
            low_senkou_b[i] = np.min(low[i - period_senkou_b + 1:i + 1])
    senkou_b = (high_senkou_b + low_senkou_b) / 2
    
    # Align Ichimoku components to daily (no shift needed as they are already forward-looking in calculation)
    # But we need to ensure we don't use future data: the Ichimoku lines are plotted ahead, so we use values from today
    # For entry logic, we use current Tenkan, Kijun, Senkou A, Senkou B without additional shift
    
    # Weekly trend filter: Tenkan/Kijun cross and price vs Senkou Span
    # Weekly Tenkan-sen
    high_tenkan_1w = np.full_like(high_1w, np.nan)
    low_tenkan_1w = np.full_like(low_1w, np.nan)
    if len(high_1w) >= period_tenkan:
        for i in range(period_tenkan - 1, len(high_1w)):
            high_tenkan_1w[i] = np.max(high_1w[i - period_tenkan + 1:i + 1])
            low_tenkan_1w[i] = np.min(low_1w[i - period_tenkan + 1:i + 1])
    tenkan_1w = (high_tenkan_1w + low_tenkan_1w) / 2
    
    # Weekly Kijun-sen
    high_kijun_1w = np.full_like(high_1w, np.nan)
    low_kijun_1w = np.full_like(low_1w, np.nan)
    if len(high_1w) >= period_kijun:
        for i in range(period_kijun - 1, len(high_1w)):
            high_kijun_1w[i] = np.max(high_1w[i - period_kijun + 1:i + 1])
            low_kijun_1w[i] = np.min(low_1w[i - period_kijun + 1:i + 1])
    kijun_1w = (high_kijun_1w + low_kijun_1w) / 2
    
    # Weekly Senkou Span A and B
    senkou_a_1w = (tenkan_1w + kijun_1w) / 2
    high_senkou_b_1w = np.full_like(high_1w, np.nan)
    low_senkou_b_1w = np.full_like(low_1w, np.nan)
    if len(high_1w) >= period_senkou_b:
        for i in range(period_senkou_b - 1, len(high_1w)):
            high_senkou_b_1w[i] = np.max(high_1w[i - period_senkou_b + 1:i + 1])
            low_senkou_b_1w[i] = np.min(low_1w[i - period_senkou_b + 1:i + 1])
    senkou_b_1w = (high_senkou_b_1w + low_senkou_b_1w) / 2
    
    # Align weekly indicators to daily timeframe (wait for weekly close)
    tenkan_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_1w)
    kijun_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_1w)
    senkou_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_a_1w)
    senkou_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period_senkou_b, period_kijun)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or \
           np.isnan(tenkan_1w_aligned[i]) or np.isnan(kijun_1w_aligned[i]) or \
           np.isnan(senkou_a_1w_aligned[i]) or np.isnan(senkou_b_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Price breaks above Kumo (Senkou Span A) AND bullish weekly Tenkan/Kijun cross AND price > weekly Senkou Span A
            if close[i] > senkou_a[i] and tenkan_1w_aligned[i] > kijun_1w_aligned[i] and close[i] > senkou_a_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below Kumo (Senkou Span B) AND bearish weekly Tenkan/Kijun cross AND price < weekly Senkou Span B
            elif close[i] < senkou_b[i] and tenkan_1w_aligned[i] < kijun_1w_aligned[i] and close[i] < senkou_b_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below Kumo (Senkou Span B) or weekly trend turns bearish
            if close[i] < senkou_b[i] or tenkan_1w_aligned[i] < kijun_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above Kumo (Senkou Span A) or weekly trend turns bullish
            if close[i] > senkou_a[i] or tenkan_1w_aligned[i] > kijun_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals