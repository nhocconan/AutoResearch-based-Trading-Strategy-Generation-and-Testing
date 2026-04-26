#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend_VolumeConfirm
Hypothesis: Ichimoku TK cross + Kumo twist (Senkou A/B cross) from 1d as regime filter, with 6h TK cross for entry timing and volume confirmation. Works in bull/bear markets by using Kumo twist to detect trend changes early and TK cross for momentum confirmation. Designed for 50-150 total trades over 4 years (12-37/year) with discrete position sizing (0.0, ±0.25) to minimize fee churn.
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
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Load 1d data for Ichimoku (Kumo twist as regime filter)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (period52_high + period52_low) / 2
    
    # Kumo twist: Senkou A crossing above/below Senkou B (trend change signal)
    # We use the non-shifted values to detect the cross, then align with proper delay
    senkou_a_raw = senkou_a_1d
    senkou_b_raw = senkou_b_1d
    
    # Align Ichimoku components to LTF (1d values available after the 1d bar closes)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_raw)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_raw)
    
    # Calculate 6h Ichimoku for entry timing (TK cross)
    period9_high_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6h = (period9_high_6h + period9_low_6h) / 2
    
    period26_high_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6h = (period26_high_6h + period26_low_6h) / 2
    
    # Volume confirmation: volume > 1.5 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(26, 52) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Discrete position sizing
        base_size = 0.25
        
        # Kumo twist detection from 1d: Senkou A > Senkou B = bullish twist, Senkou A < Senkou B = bearish twist
        kumo_bullish = senkou_a_aligned[i] > senkou_b_aligned[i]
        kumo_bearish = senkou_a_aligned[i] < senkou_b_aligned[i]
        
        # 6h TK cross: Tenkan > Kijun = bullish, Tenkan < Kijun = bearish
        tk_bullish_6h = tenkan_6h[i] > kijun_6h[i]
        tk_bearish_6h = tenkan_6h[i] < kijun_6h[i]
        
        # Long logic: Kumo bullish twist (from 1d) + 6h TK bullish cross + volume spike
        if kumo_bullish and tk_bullish_6h and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: Kumo bearish twist (from 1d) + 6h TK bearish cross + volume spike
        elif kumo_bearish and tk_bearish_6h and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # ATR-based stoploss: exit if price moves against position by 2.5 * ATR
        elif position == 1 and close[i] < (tenkan_6h[i] + kijun_6h[i])/2 - 2.5 * atr[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > (tenkan_6h[i] + kijun_6h[i])/2 + 2.5 * atr[i]:
            signals[i] = 0.0
            position = 0
        # Exit when Kumo twist reverses or TK cross reverses
        elif position == 1 and (not kumo_bullish or not tk_bullish_6h):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not kumo_bearish or not tk_bearish_6h):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0