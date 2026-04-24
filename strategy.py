#!/usr/bin/env python3
"""
Hypothesis: 1h EMA crossover with 4h ADX trend filter and session filter (08-20 UTC).
- Uses 1h timeframe (primary) and 4h HTF for ADX trend strength confirmation.
- Entry: Fast EMA(9) crosses above/below Slow EMA(21) with 4h ADX > 25.
- Exit: Opposite EMA crossover or ADX drops below 20 (trend weakening).
- Session filter: Only trade during 08-20 UTC to avoid low-volume Asian session.
- Discrete signal size: 0.20 to minimize fee churn and manage drawdown.
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
- Works in bull/bear: ADX filter ensures we only trade strong trends, avoiding whipsaws in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate 4h ADX for trend strength filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:  # Need sufficient data for ADX calculation
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate True Range (TR)
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Calculate Directional Movement (+DM, -DM)
    up_move = high_4h[1:] - high_4h[:-1]
    down_move = low_4h[:-1] - low_4h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilder_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1]/period) + data[i]
        return result
    
    period = 14
    tr_smooth = wilder_smoothing(tr, period)
    plus_dm_smooth = wilder_smoothing(plus_dm, period)
    minus_dm_smooth = wilder_smoothing(minus_dm, period)
    
    # Calculate +DI and -DI
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = np.full_like(dx, np.nan)
    # First ADX value is simple average of first 'period' DX values
    if len(dx) >= 2*period:
        adx[2*period-1] = np.nanmean(dx[period:2*period])
        # Subsequent ADX values: smoothed = prev_adx - (prev_adx/period) + current_dx
        for i in range(2*period, len(dx)):
            adx[i] = adx[i-1] - (adx[i-1]/period) + dx[i]
    
    # Align 4h ADX to 1h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # Calculate 1h EMAs for entry signals
    ema_fast = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_slow = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate EMA crossover signals
    ema_crossover = np.zeros(n, dtype=int)  # 1 for bullish cross, -1 for bearish cross, 0 otherwise
    ema_crossover[1:] = np.where(
        (ema_fast[1:] > ema_slow[1:]) & (ema_fast[:-1] <= ema_slow[:-1]), 1,
        np.where((ema_fast[1:] < ema_slow[1:]) & (ema_fast[:-1] >= ema_slow[:-1]), -1, 0)
    )
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 2*14 + 21)  # Need ADX (2*14) and slow EMA (21)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(ema_fast[i]) or 
            np.isnan(ema_slow[i]) or np.isnan(ema_crossover[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session check
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish EMA crossover AND strong trend (ADX > 25)
            if ema_crossover[i] == 1 and adx_aligned[i] > 25:
                signals[i] = 0.20
                position = 1
            # Short: bearish EMA crossover AND strong trend (ADX > 25)
            elif ema_crossover[i] == -1 and adx_aligned[i] > 25:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: bearish EMA crossover OR trend weakening (ADX < 20)
            if ema_crossover[i] == -1 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: bullish EMA crossover OR trend weakening (ADX < 20)
            if ema_crossover[i] == 1 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA9_21_4hADX25_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0