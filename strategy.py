#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index with 1d trend filter and volume confirmation.
Long when Bull Power > 0 AND Bear Power < 0 AND 1d EMA50 > 1d EMA200 AND volume > 1.3x 20-period average.
Short when Bull Power < 0 AND Bear Power > 0 AND 1d EMA50 < 1d EMA200 AND volume > 1.3x 20-period average.
Exit when Elder Ray signals reverse (Bull Power crosses zero for longs, Bear Power crosses zero for shorts).
Uses 1d EMA crossover for trend direction (avoids counter-trend trades) and Elder Ray for momentum strength.
Target: 50-150 total trades over 4 years (12-37/year).
Elder Ray measures bull/bear power relative to EMA13; combining with 1d EMA50/200 trend filter ensures we trade with the higher timeframe trend.
"""

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
    
    # Calculate 1d EMA50 and EMA200 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d EMAs to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate Elder Ray components (13-period EMA)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(200, 13, 20)  # EMA200 (200), EMA13 (13), vol MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema50 = ema50_1d_aligned[i]
        ema200 = ema200_1d_aligned[i]
        bull = bull_power[i]
        bear = bear_power[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 AND 1d EMA50 > EMA200 AND volume spike
            if bull > 0 and bear < 0 and ema50 > ema200 and volume[i] > 1.3 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Bull Power < 0 AND Bear Power > 0 AND 1d EMA50 < EMA200 AND volume spike
            elif bull < 0 and bear > 0 and ema50 < ema200 and volume[i] > 1.3 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit when Elder Ray signals reverse
            if position == 1 and bull <= 0:  # Long exit when Bull Power crosses zero
                exit_signal = True
            elif position == -1 and bear >= 0:  # Short exit when Bear Power crosses zero
                exit_signal = True
            else:
                exit_signal = False
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_BullBearPower_1dEMA50_200_Trend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0