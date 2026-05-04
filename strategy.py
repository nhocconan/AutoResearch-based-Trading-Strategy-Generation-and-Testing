#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h EMA crossover with 1d ADX trend filter and volume confirmation
# Long when 12h EMA21 crosses above EMA55 AND 1d ADX > 25 AND volume > 1.5x 20 EMA
# Short when 12h EMA21 crosses below EMA55 AND 1d ADX > 25 AND volume > 1.5x 20 EMA
# Uses 12h primary timeframe for signal generation, 1d for trend strength (ADX) to avoid ranging markets.
# Discrete sizing (0.25) to limit fee drag. Target: 12-37 trades/year.
# Works in bull markets via trend-following longs and bear markets via trend-following shorts.

name = "12h_EMA21_55_1dADX25_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value: simple average
        result[period-1] = np.nansum(data[1:period])  # skip index 0 (nan)
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr_period = wilders_smoothing(tr, 14)
    plus_dm_period = wilders_smoothing(plus_dm, 14)
    minus_dm_period = wilders_smoothing(minus_dm, 14)
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_period / tr_period
    minus_di = 100 * minus_dm_period / tr_period
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = np.full_like(dx, np.nan)
    # First ADX: simple average of first 14 DX values
    valid_start = 14 + 13  # after first 14 DX values (index 27)
    if len(dx) > valid_start:
        adx[27] = np.nanmean(dx[14:28])  # indices 14 to 27 inclusive
        for i in range(28, len(dx)):
            if not np.isnan(adx[i-1]):
                adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align 1d ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    strong_trend = adx_aligned > 25  # ADX > 25 indicates strong trend
    
    # Calculate 12h EMA21 and EMA55
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_55 = pd.Series(close).ewm(span=55, adjust=False, min_periods=55).mean().values
    
    # EMA crossover signals
    ema_cross_up = (ema_21 > ema_55) & (np.roll(ema_21, 1) <= np.roll(ema_55, 1))
    ema_cross_down = (ema_21 < ema_55) & (np.roll(ema_21, 1) >= np.roll(ema_55, 1))
    # Handle first element
    ema_cross_up[0] = False
    ema_cross_down[0] = False
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_21[i]) or np.isnan(ema_55[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: EMA21 crosses above EMA55 AND strong trend AND volume spike
            if (ema_cross_up[i] and 
                strong_trend[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: EMA21 crosses below EMA55 AND strong trend AND volume spike
            elif (ema_cross_down[i] and 
                  strong_trend[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: EMA21 crosses below EMA55 OR trend weakens
            if (ema_cross_down[i] or 
                not strong_trend[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: EMA21 crosses above EMA55 OR trend weakens
            if (ema_cross_up[i] or 
                not strong_trend[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals