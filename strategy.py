#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly pivot points and 1d ADX for trend filtering.
# Long when price breaks above weekly R1 with 1d ADX > 25 and volume confirmation.
# Short when price breaks below weekly S1 with 1d ADX > 25 and volume confirmation.
# Exit when price returns to weekly pivot point (PP) or ADX drops below 20.
# Weekly pivot points provide key support/resistance from higher timeframe.
# ADX filters for trending conditions to avoid whipsaws in ranging markets.
# Volume confirmation ensures breakouts have institutional participation.
# Designed for low trade frequency (15-25/year) to avoid fee drag.
# Works in both bull and bear markets by following institutional breakout direction.

name = "6h_WeeklyPivot_ADX_Breakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points: PP = (H + L + C)/3
    # R1 = 2*PP - L, S1 = 2*PP - H
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    
    # Get daily data for ADX
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 14:
        return np.zeros(n)
    
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Calculate ADX (14-period)
    # True Range
    tr1 = np.abs(daily_high - daily_low)
    tr2 = np.abs(daily_high - np.concatenate([[daily_close[0]], daily_close[:-1]]))
    tr3 = np.abs(daily_low - np.concatenate([[daily_close[0]], daily_close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((daily_high - np.concatenate([[daily_high[0]], daily_high[:-1]])) > 
                       (np.concatenate([[daily_low[0]], daily_low[:-1]]) - daily_low),
                       np.maximum(daily_high - np.concatenate([[daily_high[0]], daily_high[:-1]]), 0), 0)
    dm_minus = np.where((np.concatenate([[daily_low[0]], daily_low[:-1]]) - daily_low) > 
                        (daily_high - np.concatenate([[daily_high[0]], daily_high[:-1]])),
                        np.maximum(np.concatenate([[daily_low[0]], daily_low[:-1]]) - daily_low, 0), 0)
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (period=14)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[1:period])
        # Subsequent values: smoothed = (prev * (period-1) + current) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align weekly pivot points to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_weekly, pp)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    
    # Align daily ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx)
    
    # Volume confirmation: 6h volume > 1.3x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for ADX calculation
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R1 with ADX > 25 and volume confirmation
            if (close[i] > r1_aligned[i] and 
                adx_aligned[i] > 25 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 with ADX > 25 and volume confirmation
            elif (close[i] < s1_aligned[i] and 
                  adx_aligned[i] > 25 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to PP or ADX drops below 20 (trend weakening)
            if close[i] <= pp_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to PP or ADX drops below 20 (trend weakening)
            if close[i] >= pp_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals