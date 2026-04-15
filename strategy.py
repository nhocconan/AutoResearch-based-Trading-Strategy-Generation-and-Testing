#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d pivot levels (R1/S1), volume confirmation, and ADX trend filter.
# This combination reduces overtrading by requiring trend alignment (ADX>20) and volatility breakout.
# Works in bull (breaks above R1) and bear (breaks below S1) with volume confirmation.
# Target: 15-30 trades/year to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot levels and ADX
    daily = get_htf_data(prices, '1d')
    daily_high = daily['high'].values
    daily_low = daily['low'].values
    daily_close = daily['close'].values
    
    # Calculate daily pivot levels (R1, S1)
    pivot = (daily_high + daily_low + daily_close) / 3.0
    r1 = 2 * pivot - daily_low
    s1 = 2 * pivot - daily_high
    
    # Calculate daily ADX for trend filter
    # True Range
    tr1 = np.abs(daily_high[1:] - daily_low[1:])
    tr2 = np.abs(daily_high[1:] - daily_close[:-1])
    tr3 = np.abs(daily_low[1:] - daily_close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = daily_high[1:] - daily_high[:-1]
    down_move = daily_low[:-1] - daily_low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) >= period:
            result[period-1] = np.nansum(arr[1:period])
            for i in range(period, len(arr)):
                result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    period = 14
    tr14 = wilders_smoothing(tr, period)
    plus_dm14 = wilders_smoothing(plus_dm, period)
    minus_dm14 = wilders_smoothing(minus_dm, period)
    
    # DI+ and DI-
    plus_di = np.where(tr14 != 0, (plus_dm14 / tr14) * 100, 0)
    minus_di = np.where(tr14 != 0, (minus_dm14 / tr14) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, np.abs((plus_di - minus_di) / (plus_di + minus_di)) * 100, 0)
    adx = wilders_smoothing(dx, period)
    
    # Align pivot levels and ADX to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, daily, pivot)
    r1_aligned = align_htf_to_ltf(prices, daily, r1)
    s1_aligned = align_htf_to_ltf(prices, daily, s1)
    adx_aligned = align_htf_to_ltf(prices, daily, adx)
    
    # Volume filter: current 12h volume > 1.5x 20-period average volume
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when volume filter passes and ADX > 20 (trending market)
        if volume_filter[i] and adx_aligned[i] > 20:
            # Long conditions: price breaks above R1 with volume
            if close[i] > r1_aligned[i]:
                signals[i] = 0.25
            # Short conditions: price breaks below S1 with volume
            elif close[i] < s1_aligned[i]:
                signals[i] = -0.25
            else:
                signals[i] = signals[i-1]
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_Pivot_R1_S1_Breakout_Volume_ADXFilter"
timeframe = "12h"
leverage = 1.0