#!/usr/bin/env python3
# 4h_Vortex_Trend_Confirmation_12h
# Hypothesis: Vortex indicator identifies trend direction with minimal whipsaw. 
# Long when VI+ > VI- and price > 12h EMA200 with volume confirmation.
# Short when VI- > VI+ and price < 12h EMA200 with volume confirmation.
# Uses Vortex's inherent trend strength to filter noise, reducing false signals.
# Works in bull markets (VI+ dominance in uptrend) and bear markets (VI- dominance in downtrend).
# Target: 20-50 trades/year per symbol to minimize fee drag.

name = "4h_Vortex_Trend_Confirmation_12h"
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

    # Get 12h data for Vortex calculation and trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Vortex indicator on 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original index
    
    # Vortex Indicator components
    vm_plus = np.abs(high_12h[1:] - low_12h[:-1])
    vm_minus = np.abs(low_12h[1:] - high_12h[:-1])
    vm_plus = np.concatenate([[np.nan], vm_plus])
    vm_minus = np.concatenate([[np.nan], vm_minus])
    
    # Sum over 14 periods (standard Vortex period)
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vm_plus_sum = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vm_minus_sum = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    
    vi_plus = vm_plus_sum / tr_sum
    vi_minus = vm_minus_sum / tr_sum
    
    # 12h EMA200 for trend filter
    ema200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 12h indicators to 4h timeframe
    vi_plus_aligned = align_htf_to_ltf(prices, df_12h, vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, df_12h, vi_minus)
    ema200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema200_12h)
    
    # Volume spike: volume > 2.0 * 4-period average (2 days worth at 4h)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > 2.0 * vol_ma_4
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(vi_plus_aligned[i]) or 
            np.isnan(vi_minus_aligned[i]) or 
            np.isnan(ema200_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: VI+ > VI- (bullish vortex) + price > 12h EMA200 + volume spike
            if vi_plus_aligned[i] > vi_minus_aligned[i] and close[i] > ema200_12h_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: VI- > VI+ (bearish vortex) + price < 12h EMA200 + volume spike
            elif vi_minus_aligned[i] > vi_plus_aligned[i] and close[i] < ema200_12h_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: VI- crosses above VI+ (trend change) or price < EMA200
            if vi_minus_aligned[i] > vi_plus_aligned[i] or close[i] < ema200_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: VI+ crosses above VI- (trend change) or price > EMA200
            if vi_plus_aligned[i] > vi_minus_aligned[i] or close[i] > ema200_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals