#!/usr/bin/env python3
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
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Elder Ray and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # EMA13 for Elder Ray and trend
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Elder Ray components
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Smooth Elder Ray with 6-period EMA
    bull_power_smooth = pd.Series(bull_power).ewm(span=6, adjust=False, min_periods=6).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=6, adjust=False, min_periods=6).mean().values
    
    # Volume spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(bull_power_smooth[i]) or 
            np.isnan(bear_power_smooth[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull power rising above zero, bear power negative, 1d uptrend, volume spike
            if (bull_power_smooth[i] > 0 and bear_power_smooth[i] < 0 and 
                ema_13_1d_aligned[i] > ema_13_1d_aligned[i-1] and 
                volume[i] > vol_ma_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # Short: Bear power falling below zero, bull power negative, 1d downtrend, volume spike
            elif (bear_power_smooth[i] < 0 and bull_power_smooth[i] < 0 and 
                  ema_13_1d_aligned[i] < ema_13_1d_aligned[i-1] and 
                  volume[i] > vol_ma_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bull power turns negative or bear power turns positive
            if bull_power_smooth[i] < 0 or bear_power_smooth[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bear power turns positive or bull power turns positive
            if bear_power_smooth[i] > 0 or bull_power_smooth[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Elder Ray with 1d trend filter and volume confirmation
# - Elder Ray (Bull/Bear Power) measures buying/selling pressure relative to EMA13
# - Long when Bull Power > 0 and Bear Power < 0 (buying pressure, no selling pressure)
# - Short when Bear Power < 0 and Bull Power < 0 (selling pressure, no buying pressure)
# - 1d EMA13 trend filter ensures alignment with higher timeframe trend
# - Volume confirmation (1.5x average) reduces false signals
# - Exits when power signals reverse, capturing trend exhaustion
# - Works in bull markets (buy power signals) and bear markets (sell power signals)
# - Position size 0.25 targets ~50-150 total trades over 4 years (12-37/year)
# - Elder Ray provides clear momentum signals with defined zero-line crossovers
# - 1d trend filter reduces whipsaws vs same-timeframe signals
# - Novel for 6h: Elder Ray + 1d trend + volume (not recently tried in this combination)