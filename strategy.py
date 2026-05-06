#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1w Donchian channel breakout with 1d ADX trend filter and volume confirmation
# Long when price breaks above 1w Donchian upper channel AND 1d ADX > 25 AND 4h volume > 1.5 * avg_volume(20)
# Short when price breaks below 1w Donchian lower channel AND 1d ADX > 25 AND 4h volume > 1.5 * avg_volume(20)
# Exit when price crosses 1w Donchian midpoint OR 1d ADX drops below 20 (trend weakening)
# Uses discrete sizing 0.30 to balance return and drawdown control
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# 1w Donchian provides strong structural breakout levels from weekly structure
# 1d ADX > 25 ensures we only trade strong trends, avoiding whipsaws in ranging markets
# Volume confirmation filters for institutional participation
# Works in bull markets (continuation breakouts) and bear markets (continuation breakdowns)

name = "4h_1wDonchian_1dADX25_Volume_Breakout"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for Donchian channel calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:  # Need at least 21 completed weekly bars for Donchian(20)
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1w Donchian channel (20-period)
    # Upper = max(high, lookback=20)
    # Lower = min(low, lookback=20)
    # Middle = (Upper + Lower) / 2
    high_series = pd.Series(high_1w)
    low_series = pd.Series(low_1w)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Align 1w Donchian levels to 4h timeframe (wait for completed 1w bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1w, donchian_middle)
    
    # Get 1d data ONCE before loop for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for ADX calculation
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    # True Range (TR) = max(high-low, abs(high-previous_close), abs(low-previous_close))
    # Directional Movement (+DM/-DM)
    # Smoothed TR, +DM, -DM over 14 periods
    # +DI = 100 * smoothed +DM / smoothed TR
    # -DI = 100 * smoothed -DM / smoothed TR
    # DX = 100 * abs(+DI - -DI) / (+DI + -DI)
    # ADX = smoothed DX over 14 periods
    
    # Calculate True Range
    high_low = high_1d - low_1d
    high_prev_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_prev_close = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(high_low, np.maximum(high_prev_close, low_prev_close))
    tr[0] = high_low[0]  # First period has no previous close
    
    # Calculate Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # First period has no previous values
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    def wilders_smoothing(values, period):
        """Apply Wilder's smoothing (EMA with alpha=1/period)"""
        result = np.zeros_like(values)
        result[period-1] = np.nansum(values[:period])  # First value is simple average
        for i in range(period, len(values)):
            result[i] = result[i-1] + (values[i] - result[i-1]) / period
        return result
    
    period = 14
    if len(tr) >= period:
        atr = wilders_smoothing(tr, period)
        plus_dm_smooth = wilders_smoothing(plus_dm, period)
        minus_dm_smooth = wilders_smoothing(minus_dm, period)
        
        # Avoid division by zero
        plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0.0)
        minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0.0)
        
        dx = np.where((plus_di + minus_di) != 0, 
                      100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 
                      0.0)
        adx = wilders_smoothing(dx, period)
    else:
        # Not enough data for ADX calculation
        adx = np.full_like(close_1d, np.nan)
    
    # Align 1d ADX to 4h timeframe (wait for completed 1d bar)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1w Donchian upper channel with ADX > 25 and volume spike
            if (close[i] > donchian_upper_aligned[i] and close[i-1] <= donchian_upper_aligned[i-1] and 
                adx_aligned[i] > 25 and volume_confirm[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below 1w Donchian lower channel with ADX > 25 and volume spike
            elif (close[i] < donchian_lower_aligned[i] and close[i-1] >= donchian_lower_aligned[i-1] and 
                  adx_aligned[i] > 25 and volume_confirm[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1w Donchian middle OR ADX drops below 20 (trend weakening)
            if close[i] < donchian_middle_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price crosses above 1w Donchian middle OR ADX drops below 20 (trend weakening)
            if close[i] > donchian_middle_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals