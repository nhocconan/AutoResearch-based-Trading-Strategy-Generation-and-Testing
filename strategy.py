#!/usr/bin/env python3
"""
6h_Alligator_ElderRay_TripleFilter
Hypothesis: Combines Elder Ray (Bull/Bear Power) with Williams Alligator to detect trend strength and direction on 6h timeframe. Uses 1-day EMA trend filter to align with higher timeframe momentum and volume confirmation (>1.5x average) to filter false signals. Designed for 50-150 total trades over 4 years. Works in both bull and bear markets by requiring alignment between Elder Ray, Alligator, and daily trend.
"""

name = "6h_Alligator_ElderRay_TripleFilter"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator on 6h: Jaw (13), Teeth (8), Lips (5) - all SMMA
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        sma = np.nansum(arr[:period]) / period
        result[period-1] = sma
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(high, 13)  # Alligator Jaw (blue)
    teeth = smma(high, 8)  # Alligator Teeth (red)
    lips = smma(high, 5)   # Alligator Lips (green)
    
    # Elder Ray Power: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # Volume confirmation: 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        daily_trend_up = close[i] > ema_34_1d_aligned[i]
        daily_trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_long = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_short = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:
            # Long: Bull Power positive, Alligator aligned up, volume spike, daily trend up
            if (bull_power[i] > 0 and 
                alligator_long and 
                vol_ratio[i] > 1.5 and 
                daily_trend_up):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power positive, Alligator aligned down, volume spike, daily trend down
            elif (bear_power[i] > 0 and 
                  alligator_short and 
                  vol_ratio[i] > 1.5 and 
                  daily_trend_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bear Power turns positive or Alligator alignment breaks
            if bear_power[i] > 0 or not alligator_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bull Power turns positive or Alligator alignment breaks
            if bull_power[i] > 0 or not alligator_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals