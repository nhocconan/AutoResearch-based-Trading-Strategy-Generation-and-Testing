#!/usr/bin/env python3
# 12h_Vortex_Volume_Spike
# Hypothesis: Vortex indicator identifies trend direction, with VI+ > VI- indicating uptrend and VI- > VI+ indicating downtrend.
# Enter long when VI+ crosses above VI- with volume confirmation; short when VI- crosses above VI+ with volume confirmation.
# Exit when trend reverses or volume drops below average. Uses 1d trend filter to avoid counter-trend trades.
# Vortex works well in trending markets (both bull and bear) and avoids whipsaws in ranging markets via volume filter.
# Target: 15-30 trades/year per symbol to minimize fee drag.

name = "12h_Vortex_Volume_Spike"
timeframe = "12h"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0], low[0], close[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate VM+ and VM-
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    vm_plus[0] = 0
    vm_minus[0] = 0
    
    # Calculate Vortex Indicator components (14-period)
    period = 14
    sum_tr = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    sum_vm_plus = pd.Series(vm_plus).rolling(window=period, min_periods=period).sum().values
    sum_vm_minus = pd.Series(vm_minus).rolling(window=period, min_periods=period).sum().values
    
    vi_plus = sum_vm_plus / sum_tr
    vi_minus = sum_vm_minus / sum_tr
    
    # 1d trend filter: EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * vol_ma_20
    
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
            # LONG: VI+ crosses above VI- + 1d uptrend + volume spike
            if vi_plus[i] > vi_minus[i] and vi_plus[i-1] <= vi_minus[i-1] and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: VI- crosses above VI+ + 1d downtrend + volume spike
            elif vi_minus[i] > vi_plus[i] and vi_minus[i-1] <= vi_plus[i-1] and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: VI- crosses above VI+ or trend reversal
            if vi_minus[i] > vi_plus[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: VI+ crosses above VI- or trend reversal
            if vi_plus[i] > vi_minus[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals