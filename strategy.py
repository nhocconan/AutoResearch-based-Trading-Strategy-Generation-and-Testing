#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Pivot Breakout with 1d ADX Trend Filter and Volume Confirmation
# Uses weekly Camarilla pivot levels (R4/S4 for breakout, R3/S3 for fade) from prior week.
# Long when price breaks above R4 with 1d ADX > 25 and volume spike.
# Short when price breaks below S4 with 1d ADX > 25 and volume spike.
# Weekly pivot structure provides key support/resistance levels that work in both bull and bear markets.
# ADX filter ensures we only trade in trending conditions, avoiding chop.
# Volume spike confirms institutional participation in the breakout.
# Designed for 12-30 trades/year on 6h to minimize fee drag while capturing strong moves.

name = "6h_WeeklyCamarilla_R4S4_Breakout_1dADX_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Get 1d data for ADX trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivots from prior week's OHLC
    # Using prior week's data to avoid look-ahead (current week still forming)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate pivots using prior week's data (shift by 1 to use completed week)
    weekly_high_prev = np.roll(weekly_high, 1)
    weekly_low_prev = np.roll(weekly_low, 1)
    weekly_close_prev = np.roll(weekly_close, 1)
    # Set first value to NaN since we don't have prior week
    weekly_high_prev[0] = np.nan
    weekly_low_prev[0] = np.nan
    weekly_close_prev[0] = np.nan
    
    weekly_pivot = (weekly_high_prev + weekly_low_prev + weekly_close_prev) / 3.0
    weekly_range = weekly_high_prev - weekly_low_prev
    
    # Camarilla levels
    r4 = weekly_pivot + (weekly_range * 1.1 / 2.0)
    r3 = weekly_pivot + (weekly_range * 1.1 / 4.0)
    s3 = weekly_pivot - (weekly_range * 1.1 / 4.0)
    s4 = weekly_pivot - (weekly_range * 1.1 / 2.0)
    
    # Align weekly pivot levels to 6h timeframe (using prior week's completed levels)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Calculate 1d ADX for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with original arrays
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values (using Wilder's smoothing = EMA with alpha=1/period)
    period = 14
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > R4 with ADX > 25 and volume spike
            if (close[i] > r4_aligned[i] and 
                adx_aligned[i] > 25.0 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < S4 with ADX > 25 and volume spike
            elif (close[i] < s4_aligned[i] and 
                  adx_aligned[i] > 25.0 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below R3 (profit taking) or ADX drops below 20 (trend weakening)
            if (close[i] < r3_aligned[i] or 
                adx_aligned[i] < 20.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above S3 (profit taking) or ADX drops below 20 (trend weakening)
            if (close[i] > s3_aligned[i] or 
                adx_aligned[i] < 20.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals