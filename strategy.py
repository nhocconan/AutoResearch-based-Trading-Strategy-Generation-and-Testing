#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 12h ADX Trend Filter
# Long when Bull Power > 0 AND Bear Power < 0 AND 12h ADX > 25 (strong uptrend)
# Short when Bear Power < 0 AND Bull Power < 0 AND 12h ADX > 25 (strong downtrend)
# Exit when Elder Power signals weaken OR 12h ADX < 20 (trend weakening)
# Uses discrete sizing (0.25) to limit fee drag. Target: 12-35 trades/year per symbol.
# Elder Ray measures bull/bear power via EMA13, ADX filters for trending markets only.
# Works in bull markets via longs in strong uptrends and bear markets via shorts in strong downtrends.
# Avoids ranging markets where Elder Ray gives false signals.

name = "6h_ElderRay_BullBearPower_12hADX20_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 12h data ONCE before loop for ADX calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h ADX (min_periods=14 for ADX)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.concatenate([[np.nan], close_12h[:-1]]))
    tr3 = np.abs(low_12h - np.concatenate([[np.nan], close_12h[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_12h - np.concatenate([[np.nan], high_12h[:-1]])
    down_move = np.concatenate([[np.nan], low_12h[:-1]]) - low_12h
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM (Wilder smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value: simple average
        result[period-1] = np.nanmean(values[:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(values)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_dm_smooth = wilders_smoothing(plus_dm, 14)
    minus_dm_smooth = wilders_smoothing(minus_dm, 14)
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, 14)
    
    # Align 12h ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Calculate Elder Ray on 6h data (primary timeframe)
    # EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema_13
    # Bear Power = Low - EMA13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0 AND Bear Power < 0 AND ADX > 25 (strong uptrend)
            if (bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power < 0 AND Bull Power < 0 AND ADX > 25 (strong downtrend)
            elif (bear_power[i] < 0 and 
                  bull_power[i] < 0 and 
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 OR Bear Power >= 0 OR ADX < 20 (trend weakening)
            if (bull_power[i] <= 0 or 
                bear_power[i] >= 0 or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 OR Bull Power > 0 OR ADX < 20 (trend weakening)
            if (bear_power[i] >= 0 or 
                bull_power[i] > 0 or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals