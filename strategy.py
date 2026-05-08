#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 12h ADX trend filter + volume confirmation
# Williams Alligator identifies trend direction via SMoothed Moving Averages (Jaw, Teeth, Lips).
# When Lips > Teeth > Jaw = uptrend; Lips < Teeth < Jaw = downtrend.
# 12h ADX > 25 filters for strong trends only, avoiding whipsaws in ranges.
# Volume spike confirms institutional participation.
# Works in both bull and bear markets by only trading strong trends with confirmation.
# Targets 50-150 total trades over 4 years (~12-37/year) to minimize fee drag.

name = "6h_WilliamsAlligator_12hADX_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator on 6h: SMMA(5,3), SMMA(8,5), SMMA(13,8)
    def smma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        sma = np.nansum(arr[:period]) / period
        result[period-1] = sma
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)  # SMMA(13,8)
    teeth = smma(close, 8)  # SMMA(8,5)
    lips = smma(close, 5)   # SMMA(5,3)
    
    # Get 12h data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX(14) on 12h
    plus_dm = np.zeros_like(high_12h)
    minus_dm = np.zeros_like(high_12h)
    tr = np.zeros_like(high_12h)
    
    for i in range(1, len(high_12h)):
        plus_dm[i] = max(high_12h[i] - high_12h[i-1], 0)
        minus_dm[i] = max(low_12h[i-1] - low_12h[i], 0)
        if plus_dm[i] == minus_dm[i]:
            plus_dm[i] = 0
            minus_dm[i] = 0
        tr[i] = max(
            high_12h[i] - low_12h[i],
            abs(high_12h[i] - close_12h[i-1]),
            abs(low_12h[i] - close_12h[i-1])
        )
    
    # Wilder smoothing
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    tr14 = wilder_smooth(tr, 14)
    plus_dm14 = wilder_smooth(plus_dm, 14)
    minus_dm14 = wilder_smooth(minus_dm, 14)
    
    plus_di14 = np.where(tr14 != 0, 100 * (plus_dm14 / tr14), 0)
    minus_di14 = np.where(tr14 != 0, 100 * (minus_dm14 / tr14), 0)
    
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 0)
    adx = wilder_smooth(dx, 14)
    
    adx_strong = adx > 25
    adx_weak = adx < 20
    adx_strong_aligned = align_htf_to_ltf(prices, df_12h, adx_strong)
    adx_weak_aligned = align_htf_to_ltf(prices, df_12h, adx_weak)
    
    # Volume confirmation: volume > 1.5x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (vol_ma.values * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure sufficient data for Alligator and volume
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(vol_spike[i]) or np.isnan(adx_strong_aligned[i]) or 
            np.isnan(adx_weak_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Lips > Teeth > Jaw (uptrend), volume spike, strong trend
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and vol_spike[i] and adx_strong_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Lips < Teeth < Jaw (downtrend), volume spike, strong trend
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and vol_spike[i] and adx_strong_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: trend weakens or Alligator reverses
            if adx_weak_aligned[i] or not (lips[i] > teeth[i] and teeth[i] > jaw[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: trend weakens or Alligator reverses
            if adx_weak_aligned[i] or not (lips[i] < teeth[i] and teeth[i] < jaw[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals