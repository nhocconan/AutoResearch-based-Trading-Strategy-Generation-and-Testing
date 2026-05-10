#!/usr/bin/env python3
# 6h_ElderRay_Alligator_WeeklyTrend
# Hypothesis: Combines Elder Ray (Bull/Bear Power) with Williams Alligator on 6h to capture momentum in trending markets,
# filtered by weekly trend direction (EMA50) and volume confirmation. Works in both bull and bear markets by aligning
# with higher timeframe trend, reducing whipsaws. Targets 50-150 trades over 4 years.

name = "6h_ElderRay_Alligator_WeeklyTrend"
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
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator on 6h (Jaw=13, Teeth=8, Lips=5 - all SMMA)
    def smma(arr, period):
        res = np.full_like(arr, np.nan)
        if len(arr) >= period:
            # First value is SMA
            res[period-1] = np.mean(arr[:period])
            for i in range(period, len(arr)):
                res[i] = (res[i-1] * (period-1) + arr[i]) / period
        return res
    
    jaw = smma(high, 13)  # Alligator Jaw (blue)
    teeth = smma((high + low) / 2, 8)  # Alligator Teeth (red)
    lips = smma(low, 5)   # Alligator Lips (green)
    
    # Elder Ray on 6h: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or \
           np.isnan(lips[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or \
           np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
        alligator_bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_bearish = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:
            # Long: Bull Power > 0, Alligator bullish, weekly uptrend, volume confirmation
            if bull_power[i] > 0 and alligator_bullish and close[i] > ema_50_1w_aligned[i] and volume[i] > 1.5 * vol_ma_20[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0, Alligator bearish, weekly downtrend, volume confirmation
            elif bear_power[i] < 0 and alligator_bearish and close[i] < ema_50_1w_aligned[i] and volume[i] > 1.5 * vol_ma_20[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 or Alligator turns bearish
            if bull_power[i] <= 0 or not alligator_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power >= 0 or Alligator turns bullish
            if bear_power[i] >= 0 or not alligator_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals