#!/usr/bin/env python3
"""
6h_WeeklyPivot_PriceChannelBreakout_VolumeSpike
Hypothesis: Price breaking above/below weekly pivot-derived resistance/support levels (calculated from weekly high-low-close) with volume confirmation (2x average) captures strong trending moves. Uses 1d EMA50 as trend filter to align with daily trend direction. Works in both bull and bear by following daily trend. Target: 15-35 trades/year per symbol.
"""

name = "6h_WeeklyPivot_PriceChannelBreakout_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot levels (similar to Woodie's pivot)
    # Pivot = (H + L + C) / 3
    # Resistance = 2*Pivot - L
    # Support = 2*Pivot - H
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Shift by 1 to use previous week's data
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    prev_close_1w[0] = np.nan
    
    pivot_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3
    resistance_1w = 2 * pivot_1w - prev_low_1w  # R1 equivalent
    support_1w = 2 * pivot_1w - prev_high_1w   # S1 equivalent
    
    # Align weekly pivot levels to 6h timeframe
    resistance_1w_aligned = align_htf_to_ltf(prices, df_1w, resistance_1w)
    support_1w_aligned = align_htf_to_ltf(prices, df_1w, support_1w)
    
    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: >2x 20-period average (6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA50 warmup
        if (np.isnan(resistance_1w_aligned[i]) or np.isnan(support_1w_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above weekly resistance + 1d EMA50 uptrend + volume spike
            if (close[i] > resistance_1w_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly support + 1d EMA50 downtrend + volume spike
            elif (close[i] < support_1w_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below weekly support (reversal level)
            if close[i] < support_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above weekly resistance (reversal level)
            if close[i] > resistance_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals