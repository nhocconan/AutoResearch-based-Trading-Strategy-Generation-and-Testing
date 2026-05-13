#!/usr/bin/env python3
# 4h_Vortex_Trend_With_Volume_And_Chop_Filter
# Hypothesis: Vortex Indicator identifies trend direction and strength, effective in both trending and ranging markets when combined with Chop filter to avoid false signals. 
# Long when VI+ > VI- (uptrend) with Chop > 61.8 (ranging) and volume confirmation. Short when VI- > VI+ (downtrend) with Chop > 61.8 and volume confirmation.
# Uses daily trend filter to ensure alignment with higher timeframe momentum. 
# Vortex helps catch trend changes early, Chop filter avoids whipsaws in ranging markets, volume confirms institutional participation.
# Target: 20-50 trades/year per symbol to minimize fee drag.

name = "4h_Vortex_Trend_With_Volume_And_Chop_Filter"
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

    # Get daily data for Vortex and Chop calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Vortex Indicator components
    # VM+ = |High - Prev Low|, VM- = |Low - Prev High|
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    vm_plus[0] = 0  # First value has no previous
    vm_minus[0] = 0
    
    # True Range for smoothing
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    # Smooth over 14 periods (standard Vortex period)
    vm_plus_sum = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vm_minus_sum = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    vi_plus = vm_plus_sum / tr_sum
    vi_minus = vm_minus_sum / tr_sum
    
    # Calculate Choppy Market Index (Chop) - using 14-period
    # Chop = 100 * log10(sum(TR) / (max(HH) - min(LL))) / log10(n)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    range_hl = highest_high - lowest_low
    # Avoid division by zero
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    chop = 100 * np.log10(sum_tr / range_hl) / np.log10(14)
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 4h timeframe
    vi_plus_aligned = align_htf_to_ltf(prices, df_1d, vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, df_1d, vi_minus)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: volume > 2.0 * 4-period average (approx 1 day)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > 2.0 * vol_ma_4
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(vi_plus_aligned[i]) or 
            np.isnan(vi_minus_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: VI+ > VI- (uptrend) + Chop > 61.8 (ranging) + volume spike
            if vi_plus_aligned[i] > vi_minus_aligned[i] and chop_aligned[i] > 61.8 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: VI- > VI+ (downtrend) + Chop > 61.8 (ranging) + volume spike
            elif vi_minus_aligned[i] > vi_plus_aligned[i] and chop_aligned[i] > 61.8 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: VI- > VI+ (trend reversal) or Chop < 38.2 (strong trend - avoid whipsaw)
            if vi_minus_aligned[i] > vi_plus_aligned[i] or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: VI+ > VI- (trend reversal) or Chop < 38.2 (strong trend - avoid whipsaw)
            if vi_plus_aligned[i] > vi_minus_aligned[i] or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals