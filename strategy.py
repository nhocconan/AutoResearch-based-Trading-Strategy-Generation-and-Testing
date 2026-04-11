#!/usr/bin/env python3
# 6h_1d_adx_volume_breakout_v1
# Strategy: 6h ADX + volume breakout with 1d trend filter
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: ADX > 25 indicates strong trend, volume > 2x average confirms breakout strength.
# Uses 1d EMA50 for trend filter to ensure trades align with higher timeframe direction.
# Works in both bull and bear markets by filtering for strong trending conditions only.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_adx_volume_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ADX calculation (14-period)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    def smooth(val, period):
        result = np.full_like(val, np.nan, dtype=float)
        if len(val) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(val[:period])
        # Subsequent values with Wilder's smoothing
        for i in range(period, len(val)):
            result[i] = (result[i-1] * (period-1) + val[i]) / period
        return result
    
    tr14 = smooth(tr, 14)
    plus_dm14 = smooth(plus_dm, 14)
    minus_dm14 = smooth(minus_dm, 14)
    
    # DI values
    plus_di = np.where(tr14 != 0, (plus_dm14 / tr14) * 100, 0)
    minus_di = np.where(tr14 != 0, (minus_dm14 / tr14) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 
                  np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = smooth(dx, 14)
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (2.0 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after ADX warmup
        # Skip if any required data is invalid
        if (np.isnan(adx[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(plus_di[i]) or np.isnan(minus_di[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry logic: ADX > 25 + volume confirmation + directional bias
        if (adx[i] > 25 and vol_confirm[i] and 
            plus_di[i] > minus_di[i] and uptrend and position != 1):
            position = 1
            signals[i] = 0.25
        elif (adx[i] > 25 and vol_confirm[i] and 
              minus_di[i] > plus_di[i] and downtrend and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: ADX weakens or trend changes
        elif position == 1 and (adx[i] < 20 or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (adx[i] < 20 or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals