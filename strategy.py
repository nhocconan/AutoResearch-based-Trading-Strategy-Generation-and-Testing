#!/usr/bin/env python3
"""
12h Donchian(15) Breakout + Volume Filter + 1d ADX(10) Trend Filter
Hypothesis: Shorter Donchian period increases signal frequency appropriately for 12h timeframe (target 75-200 trades over 4 years). Volume confirms breakout authenticity. 1d ADX ensures we only trade in trending markets, reducing whipsaws. Works in bull (breakouts) and bear (breakdowns) by being long/short symmetric.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian15_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load 1d data for ADX (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # ADX calculation on 1d with period=10 for faster adaptation
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
    
    period_adx = 10  # Faster adaptation for regime changes
    tr_smooth = wilder_smooth(tr, period_adx)
    dm_plus_smooth = wilder_smooth(dm_plus, period_adx)
    dm_minus_smooth = wilder_smooth(dm_minus, period_adx)
    
    # DI+ and DI-
    di_plus = np.where(tr_smooth != 0, 100 * dm_plus_smooth / tr_smooth, 0)
    di_minus = np.where(tr_smooth != 0, 100 * dm_minus_smooth / tr_smooth, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smooth(dx, period_adx)
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(15, 10)  # For Donchian(15) and ADX(10)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(adx_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Donchian channel (15-period)
        if i >= 15:
            highest_high = np.max(high[i-15:i])
            lowest_low = np.min(low[i-15:i])
        else:
            highest_high = np.max(high[:i+1]) if i > 0 else high[i]
            lowest_low = np.min(low[:i+1]) if i > 0 else low[i]
        
        # Volume filter (15-period average)
        if i >= 15:
            vol_ma = np.mean(volume[i-15:i])
            volume_filter = volume[i] > vol_ma * 1.5
        else:
            volume_filter = False
        
        # Check exits
        if position == 1:  # long position
            # Exit: price closes below Donchian lower OR ADX < 15 (weaker trend)
            if close[i] < lowest_low or adx_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR ADX < 15
            if close[i] > highest_high or adx_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + ADX trend
            bull_breakout = close[i] > highest_high
            bear_breakout = close[i] < lowest_low
            trend_filter = adx_aligned[i] > 20  # Moderate trend threshold
            
            if i >= 15 and bull_breakout and volume_filter and trend_filter:
                signals[i] = 0.25
                position = 1
            elif i >= 15 and bear_breakout and volume_filter and trend_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals