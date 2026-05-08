#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA crossover with 4h volume confirmation and 1d trend filter
# EMA(9)/EMA(21) crossover provides timely entries. 4h volume surge confirms institutional interest.
# 1d ADX > 25 ensures we only trade in strong trends, avoiding whipsaws in ranges.
# Session filter (08-20 UTC) reduces noise. Targets 15-30 trades/year to minimize fee drag.

name = "1h_EMA9_21_4hVolume_1dADX"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA(9) and EMA(21)
    ema9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # 4h volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_4h = vol_4h / vol_ma_4h
    vol_surge_4h = align_htf_to_ltf(prices, df_4h, vol_ratio_4h > 1.5)
    
    # 1d ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on daily
    tr = np.maximum(
        high_1d - low_1d,
        np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1)))
    )
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    
    # Handle first element
    tr[0] = high_1d[0] - low_1d[0]
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Wilder smoothing
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    tr14 = wilder_smooth(tr, 14)
    plus_dm14 = wilder_smooth(plus_dm, 14)
    minus_dm14 = wilder_smooth(minus_dm, 14)
    
    plus_di14 = np.where(tr14 != 0, 100 * (plus_dm14 / tr14), 0)
    minus_di14 = np.where(tr14 != 0, 100 * (minus_dm14 / tr14), 0)
    
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 0)
    adx = wilder_smooth(dx, 14)
    adx_strong = adx > 25
    adx_strong_1h = align_htf_to_ltf(prices, df_1d, adx_strong)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 20)  # EMA21 and 4h vol MA warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema9[i]) or np.isnan(ema21[i]) or 
            np.isnan(vol_surge_4h[i]) or np.isnan(adx_strong_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: EMA9 > EMA21, volume surge, strong trend, in session
            if ema9[i] > ema21[i] and vol_surge_4h[i] and adx_strong_1h[i] and in_session[i]:
                signals[i] = 0.20
                position = 1
            # Enter short: EMA9 < EMA21, volume surge, strong trend, in session
            elif ema9[i] < ema21[i] and vol_surge_4h[i] and adx_strong_1h[i] and in_session[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: EMA9 < EMA21 or trend weakens or outside session
            if ema9[i] < ema21[i] or not adx_strong_1h[i] or not in_session[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: EMA9 > EMA21 or trend weakens or outside session
            if ema9[i] > ema21[i] or not adx_strong_1h[i] or not in_session[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals