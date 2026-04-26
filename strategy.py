#!/usr/bin/env python3
"""
6h_ADX_DMI_Cross_1dTrend_Filter
Hypothesis: 6h ADX with DMI crossover signals filtered by 1d EMA50 trend. Long when +DI crosses above -DI in 1d uptrend with ADX>25. Short when -DI crosses above +DI in 1d downtrend with ADX>25. Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year. Works in both bull and bear markets by following 1d trend, avoiding counter-trend whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for ADX/DMI and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # need enough for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX and DMI on 1d data
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        """Wilder's smoothing (EMA with alpha=1/period)"""
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        # Find first valid index
        valid_start = np.where(~np.isnan(data))[0]
        if len(valid_start) == 0:
            return result
        first_idx = valid_start[0]
        result[first_idx] = data[first_idx]
        for i in range(first_idx + 1, len(data)):
            if np.isnan(data[i]):
                result[i] = result[i-1]
            else:
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    period = 14
    tr_smooth = wilders_smoothing(tr, period)
    plus_dm_smooth = wilders_smoothing(plus_dm, period)
    minus_dm_smooth = wilders_smoothing(minus_dm, period)
    
    # Calculate +DI and -DI
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, period)
    
    # Align to 6h timeframe
    plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    uptrend_1d = close_1d > ema_50_1d_aligned  # using 1d close for trend, will be aligned
    downtrend_1d = close_1d < ema_50_1d_aligned
    # Note: uptrend_1d/downtrend_1d are 1d arrays, need to align them
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d.astype(float))
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d.astype(float))
    uptrend_1d_bool = uptrend_1d_aligned > 0.5
    downtrend_1d_bool = downtrend_1d_aligned > 0.5
    
    # Crossover signals: +DI crosses above -DI (long), -DI crosses above +DI (short)
    # We need previous values to detect crossover
    plus_di_prev = np.roll(plus_di_aligned, 1)
    minus_di_prev = np.roll(minus_di_aligned, 1)
    plus_di_prev[0] = np.nan
    minus_di_prev[0] = np.nan
    
    long_signal = (plus_di_aligned > minus_di_aligned) & (plus_di_prev <= minus_di_prev)
    short_signal = (minus_di_aligned > plus_di_aligned) & (minus_di_prev <= plus_di_prev)
    
    # ADX filter: only take signals when ADX > 25 (strong trend)
    strong_trend = adx_aligned > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(30, period*2)  # need enough for ADX calculation
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(plus_di_aligned[i]) or np.isnan(minus_di_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(uptrend_1d_aligned[i]) or 
            np.isnan(downtrend_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: +DI crosses above -DI in 1d uptrend with ADX>25
            if long_signal[i] and uptrend_1d_bool[i] and strong_trend[i]:
                signals[i] = 0.25
                position = 1
            # Short: -DI crosses above +DI in 1d downtrend with ADX>25
            elif short_signal[i] and downtrend_1d_bool[i] and strong_trend[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: +DI crosses below -DI OR 1d trend changes to downtrend OR ADX falls below 20
            if (minus_di_aligned[i] > plus_di_aligned[i] or 
                not uptrend_1d_bool[i] or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: -DI crosses below +DI OR 1d trend changes to uptrend OR ADX falls below 20
            if (plus_di_aligned[i] > minus_di_aligned[i] or 
                not downtrend_1d_bool[i] or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ADX_DMI_Cross_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0