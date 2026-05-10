#!/usr/bin/env python3
# 6h_ElderRay_1dTrend_Volume
# Hypothesis: Elder Ray (Bull/Bear Power) on 6h identifies momentum extremes, filtered by 1d EMA trend and volume spikes.
# Bull Power = High - EMA13, Bear Power = Low - EMA13. Enter long when Bull Power > 0 and rising, Bear Power < 0 and falling.
# Enter short when Bear Power < 0 and falling, Bull Power < 0 and rising. 1d EMA50 filter ensures alignment with daily trend.
# Volume confirmation (current > 2.0x 20-period average) adds conviction to avoid false signals in low-volume environments.
# Designed to work in both bull and bear markets by following the higher-timeframe trend and capturing momentum shifts.
# Target: 15-35 trades/year to stay within optimal trade frequency for 6h.

name = "6h_ElderRay_1dTrend_Volume"
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
    
    # Elder Ray: EMA13 for Bull/Bear Power
    ema13_period = 13
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=ema13_period, adjust=False, min_periods=ema13_period).values
    
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 1d EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 and rising (momentum building), Bear Power < 0, 1d EMA uptrend, volume confirmation
            if (bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and bear_power[i] < 0 and
                close[i] > ema50_1d_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 and falling (momentum building), Bull Power > 0, 1d EMA downtrend, volume confirmation
            elif (bear_power[i] < 0 and bear_power[i] < bear_power[i-1] and bull_power[i] > 0 and
                  close[i] < ema50_1d_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Bull Power turns negative OR Bear Power becomes positive (momentum shift)
            if bull_power[i] <= 0 or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Bear Power turns positive OR Bull Power becomes negative (momentum shift)
            if bear_power[i] >= 0 or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals