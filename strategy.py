#!/usr/bin/env python3
"""
4h_Vortex_Trend_With_Volume_Spike
Hypothesis: Vortex Indicator identifies trend direction with less whipsaw. 
In trending markets, VI+ > VI- indicates uptrend, VI- > VI+ indicates downtrend.
Combined with volume spikes and daily trend filter to capture institutional moves.
Designed for low trade frequency (target: 20-50 trades/year) to minimize fee drag.
Works in both bull and bear regimes by following confirmed trends with volume confirmation.
"""

name = "4h_Vortex_Trend_With_Volume_Spike"
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
    
    # Get daily data for 1-day trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Vortex Indicator (VI) on 4h data
    # VM+ = |high - low_prev|, VM- = |low - high_prev|
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    vm_plus[0] = 0  # first value has no previous
    vm_minus[0] = 0
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    
    # Smooth over 14 periods
    period = 14
    vm_plus_sum = pd.Series(vm_plus).rolling(window=period, min_periods=period).sum().values
    vm_minus_sum = pd.Series(vm_minus).rolling(window=period, min_periods=period).sum().values
    tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    vi_plus = vm_plus_sum / tr_sum
    vi_minus = vm_minus_sum / tr_sum
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: VI+ > VI- (uptrend) with volume spike and above 1-day EMA34
            if (vi_plus[i] > vi_minus[i] and 
                volume_spike[i] and 
                close[i] > trend_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: VI- > VI+ (downtrend) with volume spike and below 1-day EMA34
            elif (vi_minus[i] > vi_plus[i] and 
                  volume_spike[i] and 
                  close[i] < trend_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: VI- > VI+ (trend reversal) or price drops below 1-day EMA34
            if (vi_minus[i] > vi_plus[i] or 
                close[i] < trend_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: VI+ > VI- (trend reversal) or price rises above 1-day EMA34
            if (vi_plus[i] > vi_minus[i] or 
                close[i] > trend_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals