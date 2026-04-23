#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme + 1d ADX Regime + Volume Confirmation
- Williams %R(14) < -80 = oversold (long), > -20 = overbought (short) on 6h
- 1d ADX(14) > 25 = trending regime (filter: only trade in strong trends)
- Volume confirmation (> 1.8x 20-period average) reduces false signals
- Designed for 6h timeframe to capture medium-term swings with low frequency (target: 15-30 trades/year)
- Works in both bull and bear markets by only trading when strong trend is present
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1d ADX(14) for trend regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # need enough data for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    tr[0] = high_1d[0] - low_1d[0]  # first period
    
    # Directional Movement
    up_move = pd.Series(high_1d - np.roll(high_1d, 1))
    down_move = pd.Series(np.roll(low_1d, 1) - low_1d)
    up_move = np.where(up_move < 0, 0, up_move)
    down_move = np.where(down_move < 0, 0, down_move)
    
    # +DM and -DM
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_1d = wilder_smooth(tr, 14)
    plus_di_1d = 100 * wilder_smooth(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * wilder_smooth(minus_dm, 14) / atr_1d
    
    # ADX = smoothed |+DI - -DI| / (+DI + -DI)
    dx = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    dx = np.where((plus_di_1d + minus_di_1d) == 0, 0, dx)
    adx_1d = wilder_smooth(dx, 14)
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 20, 30)  # Williams %R, vol MA, ADX calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade in strong trending regime (ADX > 25)
        if adx_1d_aligned[i] <= 25:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND volume spike
            if williams_r[i] < -80 and volume[i] > 1.8 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND volume spike
            elif williams_r[i] > -20 and volume[i] > 1.8 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R returns to neutral zone (-50) OR opposite extreme
            exit_signal = False
            if position == 1:
                # Exit long when Williams %R > -50 (recovered from oversold) or > -20 (overbought)
                if williams_r[i] > -50:
                    exit_signal = True
            elif position == -1:
                # Exit short when Williams %R < -50 (recovered from overbought) or < -80 (oversold)
                if williams_r[i] < -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dADX_Regime_VolumeConfirm"
timeframe = "6h"
leverage = 1.0