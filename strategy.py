#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Elder Ray (bull/bear power) with volume confirmation and ADX trend filter.
# Designed for low trade frequency (~20-40/year) to avoid fee drag. Uses 12h Elder Ray for momentum,
# volume surge for confirmation, and ADX to avoid choppy markets. Works in bull/bear by following
# 12h momentum with strict entry filters and ADX > 25 to ensure trending conditions.

name = "4h_ElderRay_12hVOL_ADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Elder Ray calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate EMA13 for Elder Ray
    def calculate_ema(arr, period):
        ema = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return ema
        multiplier = 2 / (period + 1)
        ema[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            ema[i] = (arr[i] - ema[i-1]) * multiplier + ema[i-1]
        return ema
    
    ema13_12h = calculate_ema(close_12h, 13)
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_12h = high_12h - ema13_12h
    bear_power_12h = low_12h - ema13_12h
    
    # Align Elder Ray components to 4h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power_12h)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power_12h)
    
    # Get daily data for ADX (trend strength)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        plus_dm = np.concatenate([[np.nan], plus_dm])
        minus_dm = np.concatenate([[np.nan], minus_dm])
        
        # Smoothed values
        def wilders_smooth(arr, period):
            smoothed = np.full_like(arr, np.nan)
            if len(arr) < period:
                return smoothed
            smoothed[period-1] = np.nanmean(arr[1:period+1])
            for i in range(period, len(arr)):
                if not np.isnan(arr[i]):
                    smoothed[i] = (smoothed[i-1] * (period-1) + arr[i]) / period
                else:
                    smoothed[i] = smoothed[i-1]
            return smoothed
        
        atr = wilders_smooth(tr, period)
        plus_di = 100 * wilders_smooth(plus_dm, period) / atr
        minus_di = 100 * wilders_smooth(minus_dm, period) / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = wilders_smooth(dx, period)
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Get 4h data for volume spike
    volume_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (volume_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: bull power positive + bear power negative (bullish) + volume spike + ADX > 25
            if bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0 and vol_spike[i] and adx_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Enter short: bull power negative + bear power positive (bearish) + volume spike + ADX > 25
            elif bull_power_aligned[i] < 0 and bear_power_aligned[i] > 0 and vol_spike[i] and adx_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bull power turns negative or bear power turns positive or ADX < 20
            if bull_power_aligned[i] <= 0 or bear_power_aligned[i] >= 0 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bull power turns positive or bear power turns negative or ADX < 20
            if bull_power_aligned[i] >= 0 or bear_power_aligned[i] <= 0 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals