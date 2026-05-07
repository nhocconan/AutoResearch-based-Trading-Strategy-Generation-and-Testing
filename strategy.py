#!/usr/bin/env python3
name = "6h_ElderRay_BullPower_BearPower_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Elder Ray components: Bull Power (High - EMA13), Bear Power (Low - EMA13) on daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Volume confirmation: spike > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.8 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13)  # Wait for volume MA and EMA
    
    for i in range(start_idx, n):
        if np.isnan(ema13_1d_aligned[i]) or np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 and rising, with volume spike in daily uptrend
            if bull_power_aligned[i] > 0 and bull_power_aligned[i] > bull_power_aligned[i-1] and vol_spike[i] and close[i] > ema13_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 and falling, with volume spike in daily downtrend
            elif bear_power_aligned[i] < 0 and bear_power_aligned[i] < bear_power_aligned[i-1] and vol_spike[i] and close[i] < ema13_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bull Power <= 0 or trend turns down
            if bull_power_aligned[i] <= 0 or close[i] < ema13_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bear Power >= 0 or trend turns up
            if bear_power_aligned[i] >= 0 or close[i] > ema13_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Elder Ray (Bull/Bear Power) on 6h with daily trend filter and volume spike captures institutional accumulation/distribution.
# Long when Bull Power (High - EMA13) is positive and rising with volume confirmation in daily uptrend.
# Short when Bear Power (Low - EMA13) is negative and falling with volume confirmation in daily downtrend.
# Works in bull markets (rising Bull Power in uptrend) and bear markets (falling Bear Power in downtrend).
# Volume spike (>1.8x average) ensures conviction behind the move.
# Designed for 6h timeframe to target 50-150 total trades over 4 years, avoiding overtrading.