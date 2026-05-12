#!/usr/bin/env python3
# 6h_Ichimoku_TK_Cross_CloudFilter_1wTrend
# Hypothesis: On 6h timeframe, enter long when Tenkan-sen crosses above Kijun-sen and price is above weekly Kumo (cloud),
# with weekly trend confirmation (price > weekly EMA50). Enter short on opposite conditions.
# Uses Ichimoku components calculated on daily data for stability, aligned to 6h.
# Weekly trend filter prevents counter-trend trades in strong trends.
# Targets 15-30 trades/year for low fee drag, works in bull via cloud breaks and in bear via trend-following shorts.

name = "6h_Ichimoku_TK_Cross_CloudFilter_1wTrend"
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
    
    # Load daily data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(daily_high).rolling(window=tenkan_period, min_periods=tenkan_period).max() +
                  pd.Series(daily_low).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(daily_high).rolling(window=kijun_period, min_periods=kijun_period).max() +
                 pd.Series(daily_low).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = (pd.Series(daily_high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() +
                     pd.Series(daily_low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2
    
    # Weekly data for trend filter and cloud ahead projection
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    
    # Calculate Kumo (cloud) boundaries - Senkou Span A/B shifted forward by 26 periods
    # For cloud ahead, we use current values (they are already plotted 26 periods ahead)
    # Kumo top = max(Senkou A, Senkou B), Kumo bottom = min(Senkou A, Senkou B)
    kumo_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    kumo_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or 
            np.isnan(weekly_ema50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        tenkan = tenkan_sen_aligned[i]
        kijun = kijun_sen_aligned[i]
        price = close[i]
        weekly_ema = weekly_ema50_aligned[i]
        kumo_top_val = kumo_top[i]
        kumo_bottom_val = kumo_bottom[i]
        
        # Determine if price is above or below cloud
        above_cloud = price > kumo_top_val
        below_cloud = price < kumo_bottom_val
        
        # TK cross signals
        tk_cross_up = tenkan > kijun and tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1]
        tk_cross_down = tenkan < kijun and tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1]
        
        if position == 0:
            # LONG: TK cross up + price above cloud + weekly uptrend
            if tk_cross_up and above_cloud and price > weekly_ema:
                signals[i] = 0.25
                position = 1
            # SHORT: TK cross down + price below cloud + weekly downtrend
            elif tk_cross_down and below_cloud and price < weekly_ema:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TK cross down OR price falls below cloud
            if tk_cross_down or price < kumo_bottom_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TK cross up OR price rises above cloud
            if tk_cross_up or price > kumo_top_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals