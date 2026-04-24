#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 Breakout with 1d ADX Trend Filter and Volume Confirmation.
- Primary timeframe: 6h for execution, HTF: 1d for ADX trend filter and Camarilla levels.
- Entry: Price breaks above Camarilla H3 (long) or below L3 (short) on 6h close, with volume > 2.0x 20-period volume MA.
- Direction filter: only long when 1d ADX(14) > 25 and +DI > -DI (uptrend), only short when 1d ADX(14) > 25 and -DI > +DI (downtrend).
- Camarilla levels from 1d provide strong support/resistance; ADX ensures trend strength.
- Volume confirmation reduces false breakouts.
- Exit: Price returns to Camarilla Pivot Point (PP) or ADX weakens (< 20) or DI crossover reverses.
- Discrete signal size: 0.25 to balance return and drawdown control.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d ADX(14) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], tr])
    
    # Directional Movement
    up_move = np.concatenate([[0], high_1d[1:] - high_1d[:-1]])
    down_move = np.concatenate([[0], low_1d[:-1] - low_1d[1:]])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    tr_smoothed = np.zeros_like(tr)
    plus_dm_smoothed = np.zeros_like(plus_dm)
    minus_dm_smoothed = np.zeros_like(minus_dm)
    
    tr_smoothed[0] = tr[0]
    plus_dm_smoothed[0] = plus_dm[0]
    minus_dm_smoothed[0] = minus_dm[0]
    
    for i in range(1, len(tr)):
        tr_smoothed[i] = tr_smoothed[i-1] + alpha * (tr[i] - tr_smoothed[i-1])
        plus_dm_smoothed[i] = plus_dm_smoothed[i-1] + alpha * (plus_dm[i] - plus_dm_smoothed[i-1])
        minus_dm_smoothed[i] = minus_dm_smoothed[i-1] + alpha * (minus_dm[i] - minus_dm_smoothed[i-1])
    
    # DI and ADX
    plus_di = 100 * plus_dm_smoothed / (tr_smoothed + 1e-10)
    minus_di = 100 * minus_dm_smoothed / (tr_smoothed + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    # Wilder's smoothing for ADX
    adx = np.zeros_like(dx)
    adx[period-1] = np.mean(dx[:period])  # First ADX value is average of first 'period' DX values
    for i in range(period, len(dx)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    # Align ADX, +DI, -DI to 6h timeframe (completed 1d bar only)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di)
    
    # Calculate Camarilla levels from 1d OHLC (use previous day's data)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift OHLC by 1 to use previous day's data (avoid look-ahead)
    high_1d_shifted = np.roll(high_1d, 1)
    low_1d_shifted = np.roll(low_1d, 1)
    close_1d_shifted = np.roll(close_1d, 1)
    # First bar: use same day's data (no prior day available)
    high_1d_shifted[0] = high_1d[0]
    low_1d_shifted[0] = low_1d[0]
    close_1d_shifted[0] = close_1d[0]
    
    # Camarilla calculations: based on previous day's range
    rng = high_1d_shifted - low_1d_shifted
    camarilla_pp = (high_1d_shifted + low_1d_shifted + close_1d_shifted) / 3
    camarilla_h3 = camarilla_pp + 1.1 * (high_1d_shifted - low_1d_shifted) / 2
    camarilla_l3 = camarilla_pp - 1.1 * (high_1d_shifted - low_1d_shifted) / 2
    
    # Align Camarilla levels to 6h timeframe (completed 1d bar only)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 20) + 1  # Need 1d ADX(14), volume MA(20), plus 1 for Camarilla shift
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(plus_di_aligned[i]) or np.isnan(minus_di_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Camarilla H3 with volume spike AND strong uptrend (ADX>25 and +DI>-DI)
            if (close[i] > camarilla_h3_aligned[i] and volume_spike[i] and 
                adx_aligned[i] > 25 and plus_di_aligned[i] > minus_di_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla L3 with volume spike AND strong downtrend (ADX>25 and -DI>+DI)
            elif (close[i] < camarilla_l3_aligned[i] and volume_spike[i] and 
                  adx_aligned[i] > 25 and minus_di_aligned[i] > plus_di_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price returns to Camarilla Pivot Point OR trend weakens (ADX<20) OR DI crossover reverses
            if (close[i] < camarilla_pp_aligned[i] or adx_aligned[i] < 20 or plus_di_aligned[i] < minus_di_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price returns to Camarilla Pivot Point OR trend weakens (ADX<20) OR DI crossover reverses
            if (close[i] > camarilla_pp_aligned[i] or adx_aligned[i] < 20 or minus_di_aligned[i] < plus_di_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Breakout_1dADX_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0