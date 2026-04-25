#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend_VolumeConfirm
Hypothesis: On 6h timeframe, Ichimoku Kumo (cloud) twist from 1d timeframe combined with 6h TK cross and volume confirmation captures trend reversals with low frequency. The Kumo twist (senkou span A/B cross) indicates major trend change, TK cross provides entry timing, and volume confirms institutional participation. Works in both bull and bear markets via trend filter from 1d. Targets 12-30 trades/year.
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
    
    # Get 1d data for HTF Ichimoku and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # need 26*2 for senkou span
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku calculations on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou_span = np.concatenate([np.full(26, np.nan), close_1d[:-26]]) if len(close_1d) > 26 else np.full(len(close_1d), np.nan)
    
    # Align Ichimoku components to 6h timeframe (wait for completed 1d bar)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    chikou_aligned = align_htf_to_ltf(prices, df_1d, chikou_span)
    
    # Kumo twist detection: senkou_a crosses senkou_b (completed 1d bar)
    # Bullish twist: senkou_a crosses above senkou_b
    # Bearish twist: senkou_a crosses below senkou_b
    senkou_a_prev = np.concatenate([[np.nan], senkou_a_aligned[:-1]])
    senkou_b_prev = np.concatenate([[np.nan], senkou_b_aligned[:-1]])
    
    bullish_twist = (senkou_a_aligned > senkou_b_aligned) & (senkou_a_prev <= senkou_b_prev)
    bearish_twist = (senkou_a_aligned < senkou_b_aligned) & (senkou_a_prev >= senkou_b_prev)
    
    # TK cross on 6h for entry timing
    # Tenkan-sen (6h): (9-period high + 9-period low)/2
    period9_high_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6h = (period9_high_6h + period9_low_6h) / 2
    
    # Kijun-sen (6h): (26-period high + 26-period low)/2
    period26_high_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6h = (period26_high_6h + period26_low_6h) / 2
    
    # TK cross signals
    tenkan_prev = np.concatenate([[np.nan], tenkan_6h[:-1]])
    kijun_prev = np.concatenate([[np.nan], kijun_6h[:-1]])
    
    tk_bullish = (tenkan_6h > kijun_6h) & (tenkan_prev <= kijun_prev)
    tk_bearish = (tenkan_6h < kijun_6h) & (tenkan_prev >= kijun_prev)
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = max(52, 26, 20)  # Ichimoku 52, TK 26, vol MA 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Look for entry signals: Kumo twist + TK cross + volume confirmation
            # Long: bullish Kumo twist + bullish TK cross + volume spike
            long_signal = bullish_twist[i] and tk_bullish[i] and volume_spike[i]
            # Short: bearish Kumo twist + bearish TK cross + volume spike
            short_signal = bearish_twist[i] and tk_bearish[i] and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Bearish TK cross (exit long)
            if tk_bearish[i]:
                signals[i] = 0.0
                position = 0
            # 2. Price closes below Kumo (exit long)
            elif close[i] < senkou_a_aligned[i] and close[i] < senkou_b_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Bullish TK cross (exit short)
            if tk_bullish[i]:
                signals[i] = 0.0
                position = 0
            # 2. Price closes above Kumo (exit short)
            elif close[i] > senkou_a_aligned[i] and close[i] > senkou_b_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0