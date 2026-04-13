#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d ADX regime filter
    # Bull Power = High - EMA13, Bear Power = EMA13 - Low
    # Long: Bull Power > 0 AND ADX > 25 (trending) AND EMA13 rising
    # Short: Bear Power > 0 AND ADX > 25 (trending) AND EMA13 falling
    # Exit: Opposite power signal or ADX < 20 (range) or EMA13 direction change
    # Uses 6h primary timeframe for balance of frequency and reliability.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[1:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
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
    
    # Align daily ADX to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h EMA13 for Elder Ray
    def ema(data, period, min_periods=None):
        if min_periods is None:
            min_periods = period
        result = np.full(len(data), np.nan)
        if len(data) < min_periods:
            return result
        multiplier = 2 / (period + 1)
        result[min_periods-1] = np.mean(data[:min_periods])
        for i in range(min_periods, len(data)):
            result[i] = (data[i] - result[i-1]) * multiplier + result[i-1]
        return result
    
    ema13 = ema(close, 13, min_periods=13)
    
    # Elder Ray: Bull Power and Bear Power
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # EMA13 direction (1 = rising, -1 = falling)
    ema13_dir = np.zeros(n)
    for i in range(1, n):
        if ema13[i] > ema13[i-1]:
            ema13_dir[i] = 1
        elif ema13[i] < ema13[i-1]:
            ema13_dir[i] = -1
        else:
            ema13_dir[i] = ema13_dir[i-1]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(ema13_dir[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: ADX > 25 = trending market
        trending = adx_aligned[i] > 25
        ranging = adx_aligned[i] < 20  # Exit condition for ranging
        
        # Entry logic
        long_entry = (bull_power[i] > 0) and trending and (ema13_dir[i] == 1)
        short_entry = (bear_power[i] > 0) and trending and (ema13_dir[i] == -1)
        
        # Exit logic
        long_exit = (bull_power[i] <= 0) or ranging or (ema13_dir[i] == -1)
        short_exit = (bear_power[i] <= 0) or ranging or (ema13_dir[i] == 1)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_elder_ray_adx_v1"
timeframe = "6h"
leverage = 1.0