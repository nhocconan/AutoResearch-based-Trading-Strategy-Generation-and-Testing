#!/usr/bin/env python3
"""
6h_ADX_Donchian_Breakout_12hTrend_Filter
Hypothesis: Donchian(20) breakout on 6h timeframe with ADX trend filter from 12h timeframe.
Only trade breakouts when 12h ADX > 25 (trending market). Uses discrete position sizing (0.25) 
to minimize fee churn. Designed for low trade frequency (~15-25/year) to work in both bull 
and bear markets by avoiding choppy/range-bound conditions via ADX filter. Breakouts in 
strong trends have higher follow-through and lower false signals.
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
    
    # Get 12h data for HTF ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX(14) on 12h data
    period = 14
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(values[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    tr_smoothed = wilders_smoothing(tr, period)
    plus_dm_smoothed = wilders_smoothing(plus_dm, period)
    minus_dm_smoothed = wilders_smoothing(minus_dm, period)
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smoothed / tr_smoothed
    minus_di = 100 * minus_dm_smoothed / tr_smoothed
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = np.full_like(dx, np.nan)
    # ADX is Wilder's smoothing of DX
    valid_dx = ~np.isnan(dx)
    if np.sum(valid_dx) >= period:
        first_valid = np.where(valid_dx)[0][0]
        adx[first_valid + period - 1] = np.nanmean(dx[first_valid:first_valid + period])
        for i in range(first_valid + period, len(dx)):
            if not np.isnan(dx[i]):
                adx[i] = adx[i-1] - (adx[i-1] / period) + dx[i]
    
    # Align HTF ADX to 6h timeframe (standard 1-bar delay for ADX)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx, additional_delay_bars=1)
    
    # Calculate Donchian channels on 6h data (20-period)
    lookback = 20
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    
    for i in range(lookback-1, len(high)):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian (20) and ADX
    start_idx = max(lookback, 30)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for Donchian breakout signals with ADX trend filter
            # Long: price breaks above Donchian upper band in trending market (ADX > 25)
            # Short: price breaks below Donchian lower band in trending market (ADX > 25)
            long_signal = (close[i] > highest_high[i]) and (adx_aligned[i] > 25)
            short_signal = (close[i] < lowest_low[i]) and (adx_aligned[i] > 25)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below Donchian middle (mean reversion) or trend weakens
            middle = (highest_high[i] + lowest_low[i]) / 2
            exit_signal = (close[i] < middle) or (adx_aligned[i] < 20)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above Donchian middle or trend weakens
            middle = (highest_high[i] + lowest_low[i]) / 2
            exit_signal = (close[i] > middle) or (adx_aligned[i] < 20)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ADX_Donchian_Breakout_12hTrend_Filter"
timeframe = "6h"
leverage = 1.0