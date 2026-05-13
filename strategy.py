#!/usr/bin/env python3
# 4h_Vortex_Volume_Trend_Filter
# Hypothesis: Use Vortex Indicator for trend direction (VI+ > VI-) combined with volume surge and 1d trend filter.
# Enter long when VI+ crosses above VI- with volume > 1.5x 20-period average and price > 1d EMA34.
# Enter short when VI- crosses above VI+ with volume > 1.5x 20-period average and price < 1d EMA34.
# Exit when Vortex signal reverses. This captures strong trends with volume confirmation, works in bull/bear via trend filter.
# Target: 20-30 trades/year on 4h to minimize fee drag.

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

    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # Calculate Vortex Indicator (14-period)
    tr1 = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr1])
    vm_plus = np.abs(high - low[:-1])
    vm_minus = np.abs(low - high[:-1])
    
    # Pad vm arrays to match length
    vm_plus = np.concatenate([[np.nan], vm_plus[1:]])
    vm_minus = np.concatenate([[np.nan], vm_minus[1:]])
    
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vm_plus14 = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vm_minus14 = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    
    vi_plus = vm_plus14 / tr14
    vi_minus = vm_minus14 / tr14

    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
        # Skip if any required value is NaN
        if (np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: VI+ crosses above VI- + volume surge + price > 1d EMA34
            if (vi_plus[i] > vi_minus[i] and vi_plus[i-1] <= vi_minus[i-1] and
                volume[i] > vol_avg_20[i] * 1.5 and
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: VI- crosses above VI+ + volume surge + price < 1d EMA34
            elif (vi_minus[i] > vi_plus[i] and vi_minus[i-1] <= vi_plus[i-1] and
                  volume[i] > vol_avg_20[i] * 1.5 and
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: VI- crosses above VI+ (trend reversal)
            if vi_minus[i] > vi_plus[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: VI+ crosses above VI- (trend reversal)
            if vi_plus[i] > vi_minus[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals