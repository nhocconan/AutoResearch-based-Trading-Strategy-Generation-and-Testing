#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1-day ADX trend filter and volume confirmation.
# Camarilla levels provide precise intraday support/resistance. Breakout of R3 (resistance 3) or S3 (support 3) with volume and ADX>25 indicates strong momentum.
# Works in both bull and bear markets by filtering for strong trends via ADX. Target: 20-50 trades per year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (using previous day's OHLC)
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.25*(high-low), S3 = close - 1.25*(high-low), S4 = close - 1.5*(high-low)
    # We use R3 and S3 for breakouts
    high_low = high_prev - low_prev
    r3 = close_prev + 1.25 * high_low
    s3 = close_prev - 1.25 * high_low
    
    # Calculate daily ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values (using Wilder's smoothing: alpha = 1/period)
    def _wilder_smoothing(array, period):
        if len(array) < period:
            return np.full_like(array, np.nan, dtype=float)
        result = np.full_like(array, np.nan, dtype=float)
        # First value is simple average
        result[period-1] = np.mean(array[:period])
        # Wilder smoothing: today = (yesterday * (period-1) + today) / period
        for i in range(period, len(array)):
            result[i] = (result[i-1] * (period-1) + array[i]) / period
        return result
    
    atr = _wilder_smoothing(tr, 14)
    plus_di_smoothed = _wilder_smoothing(plus_dm, 14)
    minus_di_smoothed = _wilder_smoothing(minus_dm, 14)
    
    # DI values
    plus_di = np.where(atr != 0, plus_di_smoothed / atr * 100, 0)
    minus_di = np.where(atr != 0, minus_di_smoothed / atr * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = _wilder_smoothing(dx, 14)
    
    # Align Camarilla and ADX to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume filter: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Breakout conditions
        long_breakout = close[i] > r3_aligned[i]
        short_breakout = close[i] < s3_aligned[i]
        
        # Entry conditions with volume confirmation
        long_entry = strong_trend and long_breakout and volume_filter[i]
        short_entry = strong_trend and short_breakout and volume_filter[i]
        
        # Exit conditions: when trend weakens or price returns to midpoint
        midpoint = (r3_aligned[i] + s3_aligned[i]) / 2
        long_exit = (not strong_trend) or (position == 1 and close[i] < midpoint)
        short_exit = (not strong_trend) or (position == -1 and close[i] > midpoint)
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dADX_TrendFilter_Volume"
timeframe = "4h"
leverage = 1.0