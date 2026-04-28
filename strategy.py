#!/usr/bin/env python3
# Hypothesis: 12h Camarilla Pivot R3/S3 Breakout with 1-day ADX trend filter and volume confirmation.
# Camarilla pivot levels provide strong support/resistance based on previous day's price action.
# Breakout above R3 or below S3 indicates institutional participation and strong momentum.
# ADX > 25 filters for trending markets, avoiding false breakouts in ranging conditions.
# Volume confirmation ensures breakouts have sufficient participation.
# Designed for 12h timeframe to target 50-150 total trades over 4 years (12-37/year).
# Works in both bull and bear markets by filtering for strong trends via ADX.

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
    
    # Get daily data for Camarilla pivots and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shift by 1 to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # First day has no previous day
    prev_high[0] = prev_low[0] = prev_close[0] = 0
    
    # Camarilla calculations
    range_ = prev_high - prev_low
    # Avoid division by zero
    range_ = np.where(range_ == 0, 1e-10, range_)
    
    # Camarilla levels
    R3 = prev_close + range_ * 1.1 / 4
    S3 = prev_close - range_ * 1.1 / 4
    
    # Calculate daily ADX (14-period) for trend filter
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Wilder's smoothing: SMMA = prev * (1 - 1/period) + current * (1/period)
        alpha = 1.0 / period
        for i in range(period, len(data)):
            result[i] = result[i-1] * (1 - alpha) + data[i] * alpha
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_di_smoothed = wilders_smoothing(plus_dm, 14)
    minus_di_smoothed = wilders_smoothing(minus_dm, 14)
    
    # DI values
    plus_di = np.where(atr != 0, plus_di_smoothed / atr * 100, 0)
    minus_di = np.where(atr != 0, minus_di_smoothed / atr * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align Camarilla levels and ADX to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume filter: volume > 1.3x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30)  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or invalid
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Breakout conditions
        breakout_long = close[i] > R3_aligned[i]
        breakout_short = close[i] < S3_aligned[i]
        
        # Entry conditions with volume confirmation
        long_entry = strong_trend and breakout_long and volume_filter[i]
        short_entry = strong_trend and breakout_short and volume_filter[i]
        
        # Exit conditions: when trend weakens or price returns to pivot area
        # Exit when trend weakens or price crosses back below/above the pivot point
        pivot_point = (df_1d['high'].iloc[-1] + df_1d['low'].iloc[-1] + df_1d['close'].iloc[-1]) / 3 if len(df_1d) > 0 else close[i]
        # Simplified exit: price returns to previous day's close area
        prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
        long_exit = (not strong_trend) or (close[i] < prev_close_aligned[i]) or (position == 1 and close[i] < S3_aligned[i])
        short_exit = (not strong_trend) or (close[i] > prev_close_aligned[i]) or (position == -1 and close[i] > R3_aligned[i])
        
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

name = "12h_Camarilla_R3S3_Breakout_1dADX_TrendFilter_Volume"
timeframe = "12h"
leverage = 1.0