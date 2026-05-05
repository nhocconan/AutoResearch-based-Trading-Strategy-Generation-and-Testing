#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power + 1d ADX Trend Filter
# Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
# Long when Bull Power > 0 AND Bear Power < 0 AND 1d ADX > 25 (trending market)
# Short when Bull Power < 0 AND Bear Power > 0 AND 1d ADX > 25 (trending market)
# Exit when power diverges from price or ADX < 20 (range market)
# Uses discrete sizing (0.25) to limit fee drag. Target: 12-30 trades/year per symbol.
# Elder Ray measures bull/bear strength relative to EMA, ADX filters for trending conditions only.
# Works in bull markets via strong bull power and bear markets via strong bear power.

name = "6h_ElderRay_Power_1dADX_Trend_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data ONCE before loop for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])
    down_move = np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(values[:period])
        # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1]/period) + values[i]
        return result
    
    period = 14
    tr_period = wilders_smoothing(tr, period)
    plus_dm_period = wilders_smoothing(plus_dm, period)
    minus_dm_period = wilders_smoothing(minus_dm, period)
    
    # Avoid division by zero
    plus_di = np.where(tr_period > 0, (plus_dm_period / tr_period) * 100, 0)
    minus_di = np.where(tr_period > 0, (minus_dm_period / tr_period) * 100, 0)
    
    dx = np.where((plus_di + minus_di) > 0, 
                  np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 
                  0)
    adx = wilders_smoothing(dx, period)
    
    # Trend conditions: ADX > 25 for strong trend, ADX < 20 for ranging
    adx_strong_trend = adx > 25
    adx_range = adx < 20
    
    # Align 1d ADX conditions to 6h timeframe
    adx_strong_trend_aligned = align_htf_to_ltf(prices, df_1d, adx_strong_trend.astype(float))
    adx_range_aligned = align_htf_to_ltf(prices, df_1d, adx_range.astype(float))
    
    # Calculate Elder Ray Power on 6h data
    ema_period = 13
    if len(close) >= ema_period:
        ema_13 = pd.Series(close).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    else:
        ema_13 = np.full(n, np.nan)
    
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(adx_strong_trend_aligned[i]) or 
            np.isnan(adx_range_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0 AND Bear Power < 0 AND strong 1d uptrend
            if (bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                adx_strong_trend_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bull Power < 0 AND Bear Power > 0 AND strong 1d downtrend
            elif (bull_power[i] < 0 and 
                  bear_power[i] > 0 and 
                  adx_strong_trend_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 OR Bear Power >= 0 OR ADX < 20 (ranging)
            if (bull_power[i] <= 0 or 
                bear_power[i] >= 0 or 
                adx_range_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bull Power >= 0 OR Bear Power <= 0 OR ADX < 20 (ranging)
            if (bull_power[i] >= 0 or 
                bear_power[i] <= 0 or 
                adx_range_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals