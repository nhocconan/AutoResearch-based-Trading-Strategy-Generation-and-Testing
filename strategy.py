#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_Breakout_1wTrend_VolumeConfirm_v1
Hypothesis: Ichimoku Kumo (cloud) twist signals combined with weekly EMA50 trend filter and volume confirmation capture strong momentum shifts on 6h timeframe. Kumo twist occurs when Senkou Span A crosses Senkou Span B, indicating potential trend change. Weekly trend filter ensures alignment with higher timeframe direction, reducing counter-trend trades. Volume confirmation validates breakout strength. Designed for medium-frequency trading (target: 50-150 total trades over 4 years) to balance signal quality and fee drag in both bull and bear markets.
"""

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
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Ichimoku calculation and weekly trend
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Ichimoku components on 1d data
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Kumo twist: Senkou Span A crosses Senkou Span B
    # Bullish twist: Senkou Span A crosses above Senkou Span B
    # Bearish twist: Senkou Span A crosses below Senkou Span B
    senkou_span_a_prev = np.roll(senkou_span_a_aligned, 1)
    senkou_span_b_prev = np.roll(senkou_span_b_aligned, 1)
    senkou_span_a_prev[0] = np.nan
    senkou_span_b_prev[0] = np.nan
    
    bullish_twist = (senkou_span_a_aligned > senkou_span_b_aligned) & (senkou_span_a_prev <= senkou_span_b_prev)
    bearish_twist = (senkou_span_a_aligned < senkou_span_b_aligned) & (senkou_span_a_prev >= senkou_span_b_prev)
    
    # Weekly trend filter: EMA50 on 1w data
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume spike detection on 6h (volume > 2.0x 20-period EMA)
    volume_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (volume_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(100, 52, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or
            np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Long logic: bullish Kumo twist + price above cloud + volume spike + in uptrend
        if bullish_twist[i] and close[i] > max(senkou_span_a_aligned[i], senkou_span_b_aligned[i]) and volume_spike[i] and uptrend:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: bearish Kumo twist + price below cloud + volume spike + in downtrend
        elif bearish_twist[i] and close[i] < min(senkou_span_a_aligned[i], senkou_span_b_aligned[i]) and volume_spike[i] and downtrend:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: price crosses Kumo in opposite direction or trend weakens
        elif position == 1 and (close[i] < min(senkou_span_a_aligned[i], senkou_span_b_aligned[i]) or not uptrend):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > max(senkou_span_a_aligned[i], senkou_span_b_aligned[i]) or not downtrend):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_Breakout_1wTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0