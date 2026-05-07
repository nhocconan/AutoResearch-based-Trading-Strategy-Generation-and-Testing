#!/usr/bin/env python3
# 1D_Vortex_Trend_Reversal_Volume
# Hypothesis: 1d strategy using Vortex Indicator (VI) for trend direction and reversal signals with volume confirmation.
# Goes long when VI+ crosses above VI- (bullish reversal) with volume > 1.5x average, short when VI- crosses above VI+ (bearish reversal) with volume > 1.5x average.
# Uses weekly trend filter (weekly EMA50) to only take trades in direction of higher timeframe trend.
# Exits when opposite Vortex crossover occurs. Designed for low trade frequency (<50/year) to avoid fee drag.
# Works in both bull and bear markets by aligning with weekly trend.

name = "1D_Vortex_Trend_Reversal_Volume"
timeframe = "1d"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate Vortex Indicator (VI) on daily data
    # VM+ = |current high - previous low|
    # VM- = |current low - previous high|
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    # Set first value to 0 to avoid using future data
    vm_plus[0] = 0
    vm_minus[0] = 0
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    # Smooth VI components (14-period)
    n_period = 14
    vi_plus = pd.Series(vm_plus).rolling(window=n_period, min_periods=n_period).sum().values / \
              pd.Series(tr).rolling(window=n_period, min_periods=n_period).sum().values
    vi_minus = pd.Series(vm_minus).rolling(window=n_period, min_periods=n_period).sum().values / \
               pd.Series(tr).rolling(window=n_period, min_periods=n_period).sum().values
    
    # Weekly trend filter: EMA50 on weekly close
    weekly_close = df_1w['close'].values
    ema50_1w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: 1.5x average volume (50-period for stability)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 50)  # Ensure we have VI, weekly EMA, and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: VI+ crosses above VI- (bullish reversal) AND price above weekly EMA50 (uptrend filter) AND volume spike
            if (vi_plus[i] > vi_minus[i] and vi_plus[i-1] <= vi_minus[i-1] and  # crossover
                close[i] > ema50_1w_aligned[i] and
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: VI- crosses above VI+ (bearish reversal) AND price below weekly EMA50 (downtrend filter) AND volume spike
            elif (vi_minus[i] > vi_plus[i] and vi_minus[i-1] <= vi_plus[i-1] and  # crossover
                  close[i] < ema50_1w_aligned[i] and
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: VI- crosses above VI+ (bearish reversal)
            if vi_minus[i] > vi_plus[i] and vi_minus[i-1] <= vi_plus[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: VI+ crosses above VI- (bullish reversal)
            if vi_plus[i] > vi_minus[i] and vi_plus[i-1] <= vi_minus[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals