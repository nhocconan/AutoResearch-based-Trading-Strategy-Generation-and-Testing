#!/usr/bin/env python3
# 6h_WeeklyPivot_DonchianBreakout_Volume
# Hypothesis: Combines weekly pivot points for directional bias with 6-hour Donchian breakouts for entry timing and volume confirmation for conviction.
# Weekly pivot points are derived from the prior week's high/low/close and provide key support/resistance levels.
# In bull markets, price tends to respect pivot support and break above resistance; in bear markets, the opposite.
# Donchian breakouts capture momentum, and volume filters ensure only high-conviction trades are taken.
# Designed to work in both bull and bear markets by aligning with weekly pivot bias.
# Uses discrete position sizing (0.25) to limit drawdown and control trade frequency.

name = "6h_WeeklyPivot_DonchianBreakout_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Load weekly data for pivot points
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H + L + C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    pivot = (high_w + low_w + close_w) / 3.0
    r1 = 2 * pivot - low_w
    s1 = 2 * pivot - high_w
    r2 = pivot + (high_w - low_w)
    s2 = pivot - (high_w - low_w)
    r3 = high_w + 2 * (pivot - low_w)
    s3 = low_w - 2 * (high_w - pivot)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_w, s3)
    
    # 6h Donchian channel (20-period)
    lookback = 20
    highest = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, lookback)  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(highest[i]) or np.isnan(lowest[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Determine bias from weekly pivot: above pivot = bullish bias, below = bearish bias
        bullish_bias = close[i] > pivot_aligned[i]
        bearish_bias = close[i] < pivot_aligned[i]
        
        vol_confirm = volume_confirm[i]
        
        if position == 0:
            # LONG: Price breaks above Donchian high AND bullish bias AND volume confirmation
            if close[i] > highest[i] and bullish_bias and vol_confirm:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low AND bearish bias AND volume confirmation
            elif close[i] < lowest[i] and bearish_bias and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price returns to weekly pivot OR breaks below Donchian low
            if close[i] < pivot_aligned[i] or close[i] < lowest[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to weekly pivot OR breaks above Donchian high
            if close[i] > pivot_aligned[i] or close[i] > highest[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals