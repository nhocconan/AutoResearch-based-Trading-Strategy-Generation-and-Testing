#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 12h ADX regime filter
# Uses Elder Ray index to measure bull/bear power relative to 13-period EMA
# 12h ADX > 25 filters for trending markets only, avoiding chop
# Bullish when Bear Power < 0 and Bull Power rising (bulls gaining control)
# Bearish when Bull Power > 0 and Bear Power falling (bears gaining control)
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 50-150 total trades over 4 years = 12-37/year for 6h.
# Works in both bull and bear: ADX regime filter ensures we only trade strong trends,
# Elder Ray provides precise entry/exit based on power shifts.

name = "6h_ElderRay_12hADX_Regime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for ADX regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate 12h ADX (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first value is NaN
    
    # Directional Movement
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
    up_move = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    down_move = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    up_move = np.concatenate([[0.0], up_move])
    down_move = np.concatenate([[0.0], down_move])
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        if len(data) < period:
            return np.full_like(data, np.nan)
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        # first value is simple average
        result[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr = wilders_smoothing(tr, 14)
    dm_plus = wilders_smoothing(up_move, 14)
    dm_minus = wilders_smoothing(down_move, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Get 1d data for Elder Ray calculation (need EMA13)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate 1d EMA13
    close_1d = df_1d['close'].values
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema_13
    bear_power = low_1d - ema_13
    
    # Align Elder Ray components to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade when ADX > 25 (trending market)
        if adx_aligned[i] <= 25:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bear Power < 0 (below EMA13) AND Bull Power rising (bulls gaining control)
            if bear_power_aligned[i] < 0 and i > 100 and bull_power_aligned[i] > bull_power_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short conditions: Bull Power > 0 (above EMA13) AND Bear Power falling (bears gaining control)
            elif bull_power_aligned[i] > 0 and i > 100 and bear_power_aligned[i] < bear_power_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power turns negative OR Bear Power starts rising (bulls losing control)
            if bull_power_aligned[i] < 0 or (i > 100 and bear_power_aligned[i] > bear_power_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power turns positive OR Bull Power starts falling (bears losing control)
            if bear_power_aligned[i] > 0 or (i > 100 and bull_power_aligned[i] < bull_power_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals