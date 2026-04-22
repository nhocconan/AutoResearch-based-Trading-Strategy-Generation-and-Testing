#!/usr/bin/env python3

"""
Hypothesis: 12h Donchian channel breakout with daily ADX trend filter and volume confirmation.
Breakouts above/below 20-period Donchian channels trigger entries when daily ADX > 25
(strong trend) and volume exceeds 1.5x 20-period average. Exits when price returns to
Donchian midpoint. Designed for low trade frequency (12-37/year) by requiring trend
strength and volume confirmation. Works in both bull and bear markets by following
the daily trend direction.
"""

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
    
    # Donchian Channel (20-period) on 12h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Daily ADX for trend strength - load ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # True Range
    tr1 = daily_high - daily_low
    tr2 = np.abs(daily_high - np.roll(daily_close, 1))
    tr3 = np.abs(daily_low - np.roll(daily_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((daily_high - np.roll(daily_high, 1)) > (np.roll(daily_low, 1) - daily_low),
                       np.maximum(daily_high - np.roll(daily_high, 1), 0), 0)
    dm_minus = np.where((np.roll(daily_low, 1) - daily_low) > (daily_high - np.roll(daily_high, 1)),
                        np.maximum(np.roll(daily_low, 1) - daily_low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (14-period)
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI and DX
    di_plus = 100 * dm_plus14 / tr14
    di_minus = 100 * dm_minus14 / tr14
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend strength filter: ADX > 25
        strong_trend = adx_aligned[i] > 25
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above Donchian high + strong trend + volume spike
            if close[i] > donchian_high[i] and strong_trend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + strong trend + volume spike
            elif close[i] < donchian_low[i] and strong_trend and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to Donchian midpoint
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below midpoint
                if close[i] < donchian_mid[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above midpoint
                if close[i] > donchian_mid[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian_Breakout_DailyADX_Trend_Volume"
timeframe = "12h"
leverage = 1.0