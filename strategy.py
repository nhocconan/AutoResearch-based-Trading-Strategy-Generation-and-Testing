#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot points and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard formula)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Align daily pivot levels to 4h timeframe (use previous day's levels)
    pivot_4h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate ADX on daily data (trend strength filter)
    # TR calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_1d[0] - low_1d[0]
    
    # +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothing (Wilder's smoothing = EMA with alpha=1/period)
    def wilder_smoothing(data, period):
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        # First value is simple average
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
        # Subsequent values
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilder_smoothing(tr, 14)
    plus_di_1d = 100 * wilder_smoothing(plus_dm, 14) / np.where(atr_1d == 0, 1e-10, atr_1d)
    minus_di_1d = 100 * wilder_smoothing(minus_dm, 14) / np.where(atr_1d == 0, 1e-10, atr_1d)
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / np.where((plus_di_1d + minus_di_1d) == 0, 1e-10, (plus_di_1d + minus_di_1d))
    adx_1d = wilder_smoothing(dx_1d, 14)
    
    # Align ADX to 4h timeframe
    adx_4h = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume filter: current volume > 1.3 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 40  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_4h[i]) or np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or
            np.isnan(volume_ma20[i]) or np.isnan(adx_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.3 * volume_ma20[i])
        
        # ADX filter: only trade when trend is strong enough (ADX > 25)
        trend_filter = adx_4h[i] > 25
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume and trend filter
            if close[i] > r1_4h[i] and volume_filter and trend_filter:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S1 with volume and trend filter
            elif close[i] < s1_4h[i] and volume_filter and trend_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below pivot point (more conservative than S1)
            if close[i] < pivot_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above pivot point
            if close[i] > pivot_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DailyPivot_Breakout_Volume_ADXFilter"
timeframe = "4h"
leverage = 1.0