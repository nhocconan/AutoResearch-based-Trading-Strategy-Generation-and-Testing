#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX regime filter
# Long when Elder Bull Power > 0 AND 1d ADX > 25 (trending) AND 1d +DI > -DI
# Short when Elder Bear Power < 0 AND 1d ADX > 25 (trending) AND 1d -DI > +DI
# Uses Elder Ray to measure bull/bear power relative to 13-period EMA, ADX to filter ranging markets.
# Designed for 6h timeframe: targets 12-37 trades/year (50-150 total over 4 years) with discrete position sizing (0.25).
# Works in bull markets via longs in bullish 1d trend and bear markets via shorts in bearish 1d trend.

name = "6h_ElderRay_1dADX_TrendFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for HTF regime filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 13-period EMA for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate 1d ADX and DI
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI and ADX
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d regime indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di)
    
    # Align Elder Ray to 6h (no additional delay needed - same timeframe calculation)
    # But we need to align the EMA13 used in Elder Ray calculation
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Recalculate Elder Ray using aligned EMA13 for consistency
    bull_power_aligned = high - ema_13_aligned
    bear_power_aligned = low - ema_13_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(plus_di_aligned[i]) or 
            np.isnan(minus_di_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0 AND ADX > 25 AND +DI > -DI
            if (bull_power_aligned[i] > 0 and 
                adx_aligned[i] > 25 and 
                plus_di_aligned[i] > minus_di_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power < 0 AND ADX > 25 AND -DI > +DI
            elif (bear_power_aligned[i] < 0 and 
                  adx_aligned[i] > 25 and 
                  minus_di_aligned[i] > plus_di_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 OR ADX <= 25 OR -DI > +DI (trend weakening)
            if (bull_power_aligned[i] <= 0 or 
                adx_aligned[i] <= 25 or 
                minus_di_aligned[i] > plus_di_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 OR ADX <= 25 OR +DI > -DI (trend weakening)
            if (bear_power_aligned[i] >= 0 or 
                adx_aligned[i] <= 25 or 
                plus_di_aligned[i] > minus_di_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals