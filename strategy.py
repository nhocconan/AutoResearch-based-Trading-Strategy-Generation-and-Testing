#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 12h ADX Regime Filter
# Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# Long when Bull Power > 0 AND Bear Power < 0 AND ADX > 25 (trending market)
# Short when Bear Power > 0 AND Bull Power < 0 AND ADX > 25 (trending market)
# Exit when Elder Ray signals weaken OR ADX < 20 (range market)
# Uses 1d EMA13 for Elder Ray calculation (more stable than 6h)
# 12h ADX filters for trending conditions only, reducing whipsaws in ranging markets
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Timeframe: 6h (primary), HTF: 1d for EMA13, 12h for ADX

name = "6h_ElderRay_12hADX_Regime_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data ONCE before loop for EMA13 (Elder Ray)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 12h data ONCE before loop for ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 1d EMA(13) for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components from 1d data
    bull_power_1d = high_1d - ema_13_1d  # Bull Power = High - EMA(13)
    bear_power_1d = ema_13_1d - low_1d   # Bear Power = EMA(13) - Low
    
    # Calculate 12h ADX
    # TR = max(high-low, abs(high-previous_close), abs(low-previous_close))
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR is just high-low
    
    # +DM = max(high - previous_high, 0) but only if > previous_low - low
    up_move = high_12h - np.roll(high_12h, 1)
    down_move = np.roll(low_12h, 1) - low_12h
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(values, period):
        result = np.zeros_like(values)
        result[period-1] = np.nansum(values[:period])  # first value is simple average
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    period = 14
    atr_12h = wilders_smoothing(tr, period)
    plus_di_12h = 100 * wilders_smoothing(plus_dm, period) / atr_12h
    minus_di_12h = 100 * wilders_smoothing(minus_dm, period) / atr_12h
    dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h)
    adx_12h = wilders_smoothing(dx_12h, period)
    
    # Align HTF indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 AND ADX > 25 (strong trend)
            if (bull_power_aligned[i] > 0 and 
                bear_power_aligned[i] < 0 and 
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0 AND Bull Power < 0 AND ADX > 25 (strong trend)
            elif (bear_power_aligned[i] > 0 and 
                  bull_power_aligned[i] < 0 and 
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Elder Ray weakens OR ADX < 20 (losing trend/moving to range)
            if (bull_power_aligned[i] <= 0 or bear_power_aligned[i] >= 0 or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Elder Ray weakens OR ADX < 20 (losing trend/moving to range)
            if (bear_power_aligned[i] <= 0 or bull_power_aligned[i] >= 0 or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals