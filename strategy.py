# 4h Camarilla Pivot Breakout with Volume Spike and ADX Trend Filter
# Uses Camarilla pivot levels from daily timeframe for precision entries
# Volume spike confirms institutional participation
# ADX > 25 ensures trending market to avoid whipsaws in ranging conditions
# Designed for 4h timeframe with tight entry conditions to limit trades to 20-50/year
# Works in both bull and bear markets by following the trend defined by ADX

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (R1, S1) from previous day
    # Camarilla: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    # Using previous day's high/low/close to avoid look-ahead
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_range = prev_high - prev_low
    R1 = prev_close + camarilla_range * 1.1 / 12
    S1 = prev_close - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Load 1d data for ADX trend filter
    # ADX calculation: +DI, -DI, DX, then smoothed ADX
    period = 14
    # True Range
    tr1 = np.abs(np.roll(high_1d, 1) - np.roll(low_1d, 1))
    tr2 = np.abs(np.roll(high_1d, 1) - np.roll(close_1d, 1))
    tr3 = np.abs(np.roll(low_1d, 1) - np.roll(close_1d, 1))
    tr = np.maximum.reduce([tr1, tr2, tr3])
    tr[0] = np.nan  # First value has no previous
    
    # Directional Movement
    up_move = np.roll(high_1d, 1) - high_1d
    down_move = low_1d - np.roll(low_1d, 1)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    def smoothed_series(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: Wilder smoothing
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]):
                    result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_smoothed = smoothed_series(tr, period)
    plus_dm_smoothed = smoothed_series(plus_dm, period)
    minus_dm_smoothed = smoothed_series(minus_dm, period)
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smoothed / tr_smoothed
    minus_di = 100 * minus_dm_smoothed / tr_smoothed
    # DX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    # ADX: smoothed DX
    adx = smoothed_series(dx, period)
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume spike filter (20-period)
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma20[i] = np.mean(volume[i-20:i])
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup periods
        # Skip if data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with volume spike and ADX > 25 (uptrend)
            if close[i] > R1_aligned[i] and vol_spike[i] and adx_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume spike and ADX > 25 (downtrend)
            elif close[i] < S1_aligned[i] and vol_spike[i] and adx_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite level or trend weakens (ADX < 20)
            if position == 1:
                if close[i] < S1_aligned[i] or adx_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > R1_aligned[i] or adx_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_Breakout_Volume_ADX_v1"
timeframe = "4h"
leverage = 1.0