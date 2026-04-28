#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1-day ADX trend filter and volume confirmation.
# Donchian channels identify volatility breakouts; ADX > 25 filters for strong trends to avoid whipsaws in ranging markets.
# Volume confirmation ensures breakouts have participation. Designed for 12h timeframe to target 50-150 total trades over 4 years (12-37/year).
# Works in both bull and bear markets by filtering for strong trends via ADX and using breakout logic that captures momentum in either direction.

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
    
    # Get daily data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate daily ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
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
    
    # Smoothed values with proper smoothing (Wilder's smoothing)
    def _wilder_smooth(array, period):
        """Wilder's smoothing (equivalent to EMA with alpha=1/period)"""
        if len(array) < period:
            return np.full_like(array, np.nan, dtype=float)
        result = np.full_like(array, np.nan, dtype=float)
        # First value is simple average
        result[period-1] = np.mean(array[:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(array)):
            result[i] = (result[i-1] * (period-1) + array[i]) / period
        return result
    
    atr = _wilder_smooth(tr, 14)
    plus_di_smoothed = _wilder_smooth(plus_dm, 14)
    minus_di_smoothed = _wilder_smooth(minus_dm, 14)
    
    # DI values
    plus_di = np.where(atr != 0, plus_di_smoothed / atr * 100, 0)
    minus_di = np.where(atr != 0, minus_di_smoothed / atr * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = _wilder_smooth(dx, 14)
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Donchian channels (20-period) on 12h data
    def _rolling_max(array, window):
        """Rolling maximum"""
        if len(array) < window:
            return np.full_like(array, np.nan, dtype=float)
        result = np.full_like(array, np.nan, dtype=float)
        for i in range(window-1, len(array)):
            result[i] = np.max(array[i-window+1:i+1])
        return result
    
    def _rolling_min(array, window):
        """Rolling minimum"""
        if len(array) < window:
            return np.full_like(array, np.nan, dtype=float)
        result = np.full_like(array, np.nan, dtype=float)
        for i in range(window-1, len(array)):
            result[i] = np.min(array[i-window+1:i+1])
        return result
    
    upper_channel = _rolling_max(high, 20)
    lower_channel = _rolling_min(low, 20)
    
    # Volume filter: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Donchian breakout conditions
        breakout_up = close[i] > upper_channel[i-1]  # Break above previous upper channel
        breakout_down = close[i] < lower_channel[i-1]  # Break below previous lower channel
        
        # Entry conditions with volume confirmation and trend filter
        long_entry = strong_trend and breakout_up and volume_filter[i]
        short_entry = strong_trend and breakout_down and volume_filter[i]
        
        # Exit conditions: when trend weakens or opposite breakout occurs
        long_exit = (not strong_trend) or breakout_down
        short_exit = (not strong_trend) or breakout_up
        
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

name = "12h_DonchianBreakout_1dADX_TrendFilter_Volume"
timeframe = "12h"
leverage = 1.0