#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel breakout with 1d ADX trend filter and volume confirmation.
Long when price breaks above upper band with ADX > 25 and volume spike.
Short when price breaks below lower band with ADX > 25 and volume spike.
Exit when price returns to middle band or ADX drops below 20.
Designed for low trade frequency (15-25/year) to minimize fee drift.
Works in bull (breakouts) and bear (breakdowns) via ADX filter.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for ADX - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Calculate ADX (14)
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    close_d = df_daily['close'].values
    
    # True Range
    tr1 = high_d - low_d
    tr2 = np.abs(high_d - np.roll(close_d, 1))
    tr3 = np.abs(low_d - np.roll(close_d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = high_d - np.roll(high_d, 1)
    down_move = np.roll(low_d, 1) - low_d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    def _wilder_smooth(arr, period):
        smoothed = np.zeros_like(arr)
        smoothed[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + arr[i]
        return smoothed
    
    tr14 = _wilder_smooth(tr, 14)
    plus_dm14 = _wilder_smooth(plus_dm, 14)
    minus_dm14 = _wilder_smooth(minus_dm, 14)
    
    # Avoid division by zero
    plus_di14 = np.where(tr14 != 0, (plus_dm14 / tr14) * 100, 0)
    minus_di14 = np.where(tr14 != 0, (minus_dm14 / tr14) * 100, 0)
    
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14) * 100, 0)
    adx = _wilder_smooth(dx, 14)
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx)
    
    # Calculate Donchian channels (20-period) on 4h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    mid_20 = (high_20 + low_20) / 2.0
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper band with strong trend and volume
            if (close[i] > high_20[i] and 
                adx_aligned[i] > 25 and 
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower band with strong trend and volume
            elif (close[i] < low_20[i] and 
                  adx_aligned[i] > 25 and 
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to middle OR trend weakens
                if close[i] < mid_20[i] or adx_aligned[i] < 20:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to middle OR trend weakens
                if close[i] > mid_20[i] or adx_aligned[i] < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_ADX25_Volume"
timeframe = "4h"
leverage = 1.0
#%%