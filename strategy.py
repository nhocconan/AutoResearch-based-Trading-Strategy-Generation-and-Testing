#3o
#!/usr/bin/env python3
# Hypothesis: 12h Donchian breakout with 1-day ADX trend filter and volume confirmation.
# Donchian channels (20-period high/low) identify breakouts in trending markets.
# ADX > 25 filters for strong trends to avoid false breakouts in ranging markets.
# Volume confirmation ensures breakouts have participation.
# Designed for 12h timeframe to target 50-150 total trades over 4 years (12-37/year).
# Works in both bull and bear markets by filtering for strong trends via ADX.

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
    
    # Smoothed values
    atr = np.zeros_like(high_1d)
    plus_di_smoothed = np.zeros_like(high_1d)
    minus_di_smoothed = np.zeros_like(high_1d)
    
    # Wilder's smoothing (alpha = 1/period)
    if len(tr) >= 14:
        atr[13] = np.mean(tr[:14])
        plus_di_smoothed[13] = np.mean(plus_dm[:14])
        minus_di_smoothed[13] = np.mean(minus_dm[:14])
        for i in range(14, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
            plus_di_smoothed[i] = (plus_di_smoothed[i-1] * 13 + plus_dm[i]) / 14
            minus_di_smoothed[i] = (minus_di_smoothed[i-1] * 13 + minus_dm[i]) / 14
    
    # DI values
    plus_di = np.where(atr != 0, plus_di_smoothed / atr * 100, 0)
    minus_di = np.where(atr != 0, minus_di_smoothed / atr * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = np.zeros_like(dx)
    if len(dx) >= 14:
        adx[13] = np.mean(dx[:14])
        for i in range(14, len(dx)):
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Donchian channels (20-period) on 12h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Donchian breakout conditions
        long_breakout = close[i] > highest_high[i-1]  # Break above prior 20-period high
        short_breakout = close[i] < lowest_low[i-1]   # Break below prior 20-period low
        
        # Entry conditions with volume confirmation
        long_entry = strong_trend and long_breakout and volume_filter[i]
        short_entry = strong_trend and short_breakout and volume_filter[i]
        
        # Exit conditions: when trend weakens or opposite breakout occurs
        long_exit = (not strong_trend) or short_breakout
        short_exit = (not strong_trend) or long_breakout
        
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