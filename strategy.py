#!/usr/bin/env python3
# 4h_Vortex_T1_Trend_1d_Trend_Volume
# Hypothesis: Combine Vortex indicator (trend strength) with 1d trend and volume confirmation.
# Long: VI+ > VI- (bullish trend) + price > 1d EMA50 + volume spike.
# Short: VI- > VI+ (bearish trend) + price < 1d EMA50 + volume spike.
# Uses Vortex to capture trend direction and strength, reducing false signals in chop.
# Target: 20-50 trades/year per symbol to minimize fee drag.

name = "4h_Vortex_T1_Trend_1d_Trend_Volume"
timeframe = "4h"
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
    volume = prices['volume'].values

    # Get 1d data for EMA trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Vortex indicator on 4h data
    # VM+ = |high - low_prev|, VM- = |low - high_prev|
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    
    vm_plus = np.abs(high - low_prev)
    vm_minus = np.abs(low - high_prev)
    
    # Sum over 14 periods (standard Vortex period)
    vm_plus_sum = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vm_minus_sum = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    tr = pd.Series(np.maximum(np.abs(high - low), np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))).rolling(window=14, min_periods=14).sum().values
    
    vi_plus = vm_plus_sum / tr
    vi_minus = vm_minus_sum / tr
    
    # Volume spike: volume > 2.0 * 3-period average (1.5 days worth at 4h)
    vol_ma_3 = pd.Series(volume).rolling(window=3, min_periods=3).mean().values
    volume_spike = volume > 2.0 * vol_ma_3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(vi_plus[i]) or 
            np.isnan(vi_minus[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: VI+ > VI- (bullish vortex) + price > 1d EMA50 + volume spike
            if vi_plus[i] > vi_minus[i] and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: VI- > VI+ (bearish vortex) + price < 1d EMA50 + volume spike
            elif vi_minus[i] > vi_plus[i] and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: VI- > VI+ (trend reversal) or price < 1d EMA50
            if vi_minus[i] > vi_plus[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: VI+ > VI- (trend reversal) or price > 1d EMA50
            if vi_plus[i] > vi_minus[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals