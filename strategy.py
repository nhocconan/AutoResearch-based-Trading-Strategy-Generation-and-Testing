#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_1d_trix_volume_breakout_v1
# Uses TRIX (triple exponential smoothed momentum) on 1d timeframe to detect momentum shifts.
# Enters long when TRIX crosses above zero with volume confirmation and price above 200 EMA.
# Enters short when TRIX crosses below zero with volume confirmation and price below 200 EMA.
# Uses ADX > 20 to filter for trending conditions, avoiding range-bound whipsaws.
# Designed for low trade frequency (target: 20-50 trades/year) to minimize fee drag.
# Works in bull markets (momentum continuation) and bear markets (momentum reversals).

name = "4h_1d_trix_volume_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for TRIX and EMA200
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate TRIX on daily close: triple EMA of 1-period % change
    # TRIX = EMA(EMA(EMA(close_pct, 12), 12), 12) * 100
    close_pct = np.diff(df_1d['close'].values) / df_1d['close'].values[:-1] * 100
    close_pct = np.concatenate([[0], close_pct])  # align length
    
    def ema(arr, period):
        return pd.Series(arr).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    ema1 = ema(close_pct, 12)
    ema2 = ema(ema1, 12)
    ema3 = ema(ema2, 12)
    trix_raw = ema3  # already in % terms
    
    # Align TRIX to 4h timeframe
    trix_1d = align_htf_to_ltf(prices, df_1d, trix_raw)
    
    # Calculate 200 EMA on daily close for trend filter
    ema200_1d = ema(df_1d['close'].values, 200)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period average (4h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    # ADX trend filter: only trade when ADX > 20
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Plus and Minus Directional Movement
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    
    # Wilder's smoothing function
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smooth(tr, 14)
    plus_dm_smooth = wilders_smooth(plus_dm, 14)
    minus_dm_smooth = wilders_smooth(minus_dm, 14)
    
    # Avoid division by zero
    plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0)
    minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smooth(dx, 14)
    adx_filter = adx > 20  # trending condition
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # start after warmup
        # Skip if indicators not ready
        if np.isnan(trix_1d[i]) or np.isnan(ema200_1d_aligned[i]) or np.isnan(adx_filter[i]):
            signals[i] = 0.0
            continue
        
        # Require both volume and trend filters
        if not (vol_confirm[i] and adx_filter[i]):
            # Hold current position if filters fail
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: TRIX crosses above zero AND price above EMA200
        if trix_1d[i] > 0 and trix_1d[i-1] <= 0 and close[i] > ema200_1d_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: TRIX crosses below zero AND price below EMA200
        elif trix_1d[i] < 0 and trix_1d[i-1] >= 0 and close[i] < ema200_1d_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: TRIX crosses zero in opposite direction
        elif trix_1d[i] > 0 and trix_1d[i-1] <= 0 and position == -1:
            position = 0
            signals[i] = 0.0
        elif trix_1d[i] < 0 and trix_1d[i-1] >= 0 and position == 1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals