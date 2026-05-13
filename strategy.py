#!/usr/bin/env python3
# 4h_Vortex_Volume_Trend_Filter
# Hypothesis: The Vortex Indicator identifies trend direction by measuring upward and downward vortex movement.
# Long when VI+ > VI- with volume confirmation and price above 12h EMA50 (bullish trend).
# Short when VI- > VI+ with volume confirmation and price below 12h EMA50 (bearish trend).
# Uses 1d ADX as a regime filter: only trade when ADX > 25 (trending market).
# Designed to avoid whipsaws in ranging markets and reduce trade frequency for better generalization.

name = "4h_Vortex_Volume_Trend_Filter"
timeframe = "4h"
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

    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))

    # Vortex Indicator: +VM and -VM
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    vm_plus[0] = 0
    vm_minus[0] = 0

    # Sum over 14 periods
    period = 14
    sum_tr = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    sum_vm_plus = pd.Series(vm_plus).rolling(window=period, min_periods=period).sum().values
    sum_vm_minus = pd.Series(vm_minus).rolling(window=period, min_periods=period).sum().values

    # VI+ and VI-
    vi_plus = sum_vm_plus / sum_tr
    vi_minus = sum_vm_minus / sum_tr

    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    # ADX calculation
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    # Smoothed +/-DM
    smoothed_plus_dm = pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values
    smoothed_minus_dm = pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values
    # DI+ and DI-
    plus_di = 100 * smoothed_plus_dm / pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
    minus_di = 100 * smoothed_minus_dm / pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)

    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Volume filter: >1.8x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after warmup period for indicators
        # Skip if any required value is NaN
        if (np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: VI+ > VI- (bullish vortex) + ADX > 25 (trending) + price > EMA50 + volume spike
            if (vi_plus[i] > vi_minus[i] and 
                adx_aligned[i] > 25 and
                close[i] > ema50_12h_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.8):
                signals[i] = 0.25
                position = 1
            # SHORT: VI- > VI+ (bearish vortex) + ADX > 25 (trending) + price < EMA50 + volume spike
            elif (vi_minus[i] > vi_plus[i] and 
                  adx_aligned[i] > 25 and
                  close[i] < ema50_12h_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: VI- > VI+ or ADX drops below 20 or price breaks below EMA50
            if (vi_minus[i] > vi_plus[i] or 
                adx_aligned[i] < 20 or 
                close[i] < ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: VI+ > VI- or ADX drops below 20 or price breaks above EMA50
            if (vi_plus[i] > vi_minus[i] or 
                adx_aligned[i] < 20 or 
                close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals