#!/usr/bin/env python3
"""
4h_Vortex_Trend_Confirmation
Hypothesis: Uses 4h timeframe with Vortex indicator (VI+ and VI-) to detect trend direction,
confirmed by 1d EMA34 trend and volume surge after low volatility periods.
The Vortex indicator identifies trend initiation by comparing current high-low ranges
with prior periods, making it effective for catching momentum bursts in both bull and bear markets.
Entry occurs when VI+ crosses above VI- (bullish) or VI- crosses above VI+ (bearish),
filtered by 1d trend alignment and volume confirmation to avoid false signals.
Uses discrete position sizing (0.25) to minimize fee churn and targets 20-40 trades/year.
"""

name = "4h_Vortex_Trend_Confirmation"
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

    # Get 1d data for trend filter (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)

    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Vortex Indicator (VI+ and VI-) on 4h data
    # True Range
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # VM+ and VM-
    vm_plus = np.abs(high[1:] - low[:-1])
    vm_minus = np.abs(low[1:] - high[:-1])
    vm_plus = np.concatenate([[np.nan], vm_plus])
    vm_minus = np.concatenate([[np.nan], vm_minus])
    
    # Sum over 14 periods (standard Vortex period)
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vm_plus_sum = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vm_minus_sum = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    
    # VI+ and VI-
    vi_plus = vm_plus_sum / tr_sum
    vi_minus = vm_minus_sum / tr_sum
    
    # Volume indicators: 20-period average and volatility regime
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_std_20 = pd.Series(volume).rolling(window=20, min_periods=20).std().values
    vol_avg_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    low_vol_regime = vol_std_20 < (vol_avg_50 * 0.5)  # volatility less than half of 50-period average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start from 50 to have enough data for all indicators
        # Get aligned values for current 4h bar
        ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)[i]
        vi_plus_val = vi_plus[i]
        vi_minus_val = vi_minus[i]
        vol_avg_val = vol_avg_20[i]
        low_vol = low_vol_regime[i]
        
        # Skip if any required data is NaN
        if (np.isnan(ema34_aligned) or np.isnan(vi_plus_val) or np.isnan(vi_minus_val) or 
            np.isnan(vol_avg_val) or np.isnan(low_vol)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: VI+ crosses above VI- + 1d uptrend + low vol regime + volume spike
            if (vi_plus_val > vi_minus_val and 
                vi_plus[i-1] <= vi_minus[i-1] and  # crossover confirmation
                close[i] > ema34_aligned and 
                low_vol and 
                volume[i] > vol_avg_val * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: VI- crosses above VI+ + 1d downtrend + low vol regime + volume spike
            elif (vi_minus_val > vi_plus_val and 
                  vi_minus[i-1] <= vi_plus[i-1] and  # crossover confirmation
                  close[i] < ema34_aligned and 
                  low_vol and 
                  volume[i] > vol_avg_val * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: VI- crosses above VI+ or trend turns down
            if (vi_minus_val > vi_plus_val or close[i] < ema34_aligned):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: VI+ crosses above VI- or trend turns up
            if (vi_plus_val > vi_minus_val or close[i] > ema34_aligned):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals