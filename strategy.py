#!/usr/bin/env python3
"""
4h Donchian Breakout + Daily EMA + Volume + ADX Filter
Hypothesis: Donchian(20) breakouts on 4h with daily EMA200 trend filter, volume confirmation, and ADX trend strength filter captures institutional moves. Designed for 75-200 trades over 4 years (19-50/year) to minimize fee drag while working in both bull (breakouts) and bear (reversals at bands) markets. Uses 1d timeframe for EMA200 and ADX filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_1d_ema200_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA200 and ADX (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # EMA200 on 1d
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False).mean().values
    
    # ADX calculation on 1d (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Wilder's smoothing
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period_adx = 14
    tr_smooth = wilder_smooth(tr, period_adx)
    dm_plus_smooth = wilder_smooth(dm_plus, period_adx)
    dm_minus_smooth = wilder_smooth(dm_minus, period_adx)
    
    # DI+ and DI-
    di_plus = np.where(tr_smooth != 0, 100 * dm_plus_smooth / tr_smooth, 0)
    di_minus = np.where(tr_smooth != 0, 100 * dm_minus_smooth / tr_smooth, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smooth(dx, period_adx)
    
    # Align 1d indicators to 4h timeframe
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(20, 14)  # For Donchian and ADX
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter (20-period average)
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            volume_filter = volume[i] > vol_ma * 1.5
        else:
            volume_filter = False
        
        # Check exits: price crosses back through Donchian or ADX weakening
        if position == 1:  # long position
            # Exit: price crosses below lower Donchian OR ADX < 20
            if close[i] < lowest_low[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above upper Donchian OR ADX < 20
            if close[i] > highest_high[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: price breaks Donchian + volume + EMA200 trend + ADX trend
            bull_breakout = close[i] > highest_high[i]
            bear_breakout = close[i] < lowest_low[i]
            ema_filter_long = close[i] > ema200_1d_aligned[i]
            ema_filter_short = close[i] < ema200_1d_aligned[i]
            trend_filter = adx_aligned[i] > 25  # Strong trend
            
            if i >= 20 and bull_breakout and volume_filter and ema_filter_long and trend_filter:
                signals[i] = 0.25
                position = 1
            elif i >= 20 and bear_breakout and volume_filter and ema_filter_short and trend_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals