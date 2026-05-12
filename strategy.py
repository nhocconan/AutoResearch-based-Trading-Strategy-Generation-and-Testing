#!/usr/bin/env python3
# 6h_ElderRay_BullPower_EMA21_TrendFilter
# Hypothesis: Elder Ray Bull Power (High - EMA13) indicates bullish momentum when positive.
# Enter long when Bull Power > 0 and close > EMA21 (trend filter) with volume confirmation.
# Exit when Bull Power <= 0 (momentum fade) or close < EMA21 (trend break).
# Works in both bull and bear markets by capturing momentum shifts with trend alignment.

name = "6h_ElderRay_BullPower_EMA21_TrendFilter"
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
    
    # Calculate EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray Bull Power: High - EMA13
    bull_power = high - ema13
    
    # Trend filter: EMA21
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume confirmation: 20-period moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    
    start_idx = 30  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(bull_power[i]) or np.isnan(ema21[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        bull_power_val = bull_power[i]
        ema21_val = ema21[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # LONG: Bull Power positive (bullish momentum) + close above EMA21 (trend) + volume confirmation
            if bull_power_val > 0 and close[i] > ema21_val and volume[i] > vol_ma_val:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT: Bull Power <= 0 (momentum fade) OR close below EMA21 (trend break)
            if bull_power_val <= 0 or close[i] < ema21_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals