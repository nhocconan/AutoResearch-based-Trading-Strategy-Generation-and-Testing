#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1wADXTrend_VolumeFilter
Hypothesis: On 6h timeframe, use Donchian(20) breakouts filtered by 1-week ADX > 25 (strong trend) and volume > 1.5x 20-period average. Enter long when price breaks above upper Donchian with 1w uptrend (ADX>25 and +DI>-DI) and volume confirmation. Enter short when price breaks below lower Donchian with 1w downtrend (ADX>25 and -DI>+DI) and volume confirmation. Uses discrete position size 0.25. Designed for 12-30 trades/year by requiring strong weekly trend alignment and volume confirmation, reducing whipsaw in ranging markets while capturing momentum in both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w ADX and DI
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first value NaN
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+ and DM- using Wilder's smoothing (alpha=1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period]) if period > 1 else data[0]
        # Wilder smoothing: today = (yesterday * (period-1) + today) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    tr_smooth = wilder_smooth(tr, period)
    dm_plus_smooth = wilder_smooth(dm_plus, period)
    dm_minus_smooth = wilder_smooth(dm_minus, period)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / np.maximum(tr_smooth, 1e-10)
    di_minus = 100 * dm_minus_smooth / np.maximum(tr_smooth, 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / np.maximum(di_plus + di_minus, 1e-10)
    # ADX is Wilder smoothed DX
    adx = wilder_smooth(dx, period)
    
    # 1w trend conditions: ADX > 25 and directional bias
    adx_strong = adx > 25
    plus_di_minus_di = di_plus > di_minus  # uptrend bias
    minus_di_plus_di = di_minus > di_plus  # downtrend bias
    
    # Align 1w indicators to 6h
    adx_strong_aligned = align_htf_to_ltf(prices, df_1w, adx_strong)
    plus_di_minus_di_aligned = align_htf_to_ltf(prices, df_1w, plus_di_minus_di)
    minus_di_plus_di_aligned = align_htf_to_ltf(prices, df_1w, minus_di_plus_di)
    
    # Donchian(20) on 6h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian lookback, volume MA, and 1w ADX
    start_idx = max(lookback, 20, 2*period)  # ADX needs ~2*period for stability
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(adx_strong_aligned[i]) or
            np.isnan(plus_di_minus_di_aligned[i]) or np.isnan(minus_di_plus_di_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1w trend alignment
        trend_1w_uptrend = adx_strong_aligned[i] and plus_di_minus_di_aligned[i]
        trend_1w_downtrend = adx_strong_aligned[i] and minus_di_plus_di_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian + 1w uptrend + volume spike
            long_signal = (close[i] > highest_high[i]) and trend_1w_uptrend and volume_spike[i]
            
            # Short: price breaks below lower Donchian + 1w downtrend + volume spike
            short_signal = (close[i] < lowest_low[i]) and trend_1w_downtrend and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below lower Donchian OR 1w trend weakens/downtrend
            if (close[i] < lowest_low[i] or not trend_1w_uptrend):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above upper Donchian OR 1w trend weakens/uptrend
            if (close[i] > highest_high[i] or not trend_1w_downtrend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_1wADXTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0