#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Williams Alligator + Elder Ray Power with 1-week ADX filter.
# The Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and strength.
# Elder Ray Power measures bull/bear energy relative to EMA13.
# 1-week ADX > 25 filters for trending markets only, avoiding whipsaws in ranges.
# This combination aims for 15-25 trades per year per symbol (60-100 total over 4 years),
# staying within optimal range to minimize fee drift while capturing strong trends.
# Works in both bull (Power > 0) and bear (Power < 0) markets by following Alligator alignment.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 12h data ONCE for Alligator and Elder Ray
    df_12h = get_htf_data(prices, '12h')
    
    # Load 1w data ONCE for ADX filter
    df_1w = get_htf_data(prices, '1w')
    
    # Williams Alligator on 12h: Jaw (SMMA13), Teeth (SMMA8), Lips (SMMA5)
    # SMMA = smoothed moving average (similar to Wilder's smoothing)
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        res = np.full_like(arr, np.nan, dtype=float)
        sma = np.mean(arr[:period])
        res[period-1] = sma
        for i in range(period, len(arr)):
            res[i] = (res[i-1] * (period-1) + arr[i]) / period
        return res
    
    if len(df_12h) < 13:
        return np.zeros(n)
    
    jaw = smma(df_12h['close'].values, 13)
    teeth = smma(df_12h['close'].values, 8)
    lips = smma(df_12h['close'].values, 5)
    
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Elder Ray Power on 12h: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(df_12h['close'].values).ewm(span=13, adjust=False).mean().values
    ema13_aligned = align_htf_to_ltf(prices, df_12h, ema13)
    
    bull_power = high - ema13_aligned
    bear_power = low - ema13_aligned
    
    # ADX on 1w: measures trend strength
    def calculate_adx(high_arr, low_arr, close_arr, period=14):
        if len(high_arr) < period + 1:
            return np.full_like(high_arr, np.nan, dtype=float)
        
        # True Range
        tr1 = high_arr[1:] - low_arr[1:]
        tr2 = np.abs(high_arr[1:] - close_arr[:-1])
        tr3 = np.abs(low_arr[1:] - close_arr[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        plus_dm = np.where((high_arr[1:] - high_arr[:-1]) > (low_arr[:-1] - low_arr[1:]), 
                           np.maximum(high_arr[1:] - high_arr[:-1], 0), 0)
        minus_dm = np.where((low_arr[:-1] - low_arr[1:]) > (high_arr[1:] - high_arr[:-1]), 
                            np.maximum(low_arr[:-1] - low_arr[1:], 0), 0)
        plus_dm = np.concatenate([[np.nan], plus_dm])
        minus_dm = np.concatenate([[np.nan], minus_dm])
        
        # Smoothed TR, +DM, -DM (using Wilder's smoothing)
        def wilders_smoothing(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan, dtype=float)
            res = np.full_like(arr, np.nan, dtype=float)
            # First value is simple average
            res[period-1] = np.nansum(arr[1:period]) / (period-1) if np.sum(~np.isnan(arr[1:period])) >= (period-1) else np.nan
            for i in range(period, len(arr)):
                if np.isnan(res[i-1]) or np.isnan(arr[i]):
                    res[i] = np.nan
                else:
                    res[i] = (res[i-1] * (period-1) + arr[i]) / period
            return res
        
        tr_smoothed = wilders_smoothing(tr, period)
        plus_dm_smoothed = wilders_smoothing(plus_dm, period)
        minus_dm_smoothed = wilders_smoothing(minus_dm, period)
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smoothed / tr_smoothed
        minus_di = 100 * minus_dm_smoothed / tr_smoothed
        
        # DX and ADX
        dx = np.where((plus_di + minus_di) != 0, 
                      100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = wilders_smoothing(dx, period)
        return adx
    
    if len(df_1w) < 28:  # Need enough for ADX calculation
        return np.zeros(n)
    
    adx_1w = calculate_adx(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w, additional_delay_bars=0)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 30)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_long = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        alligator_short = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        # Elder Ray: Bull Power > 0 = bullish energy, Bear Power < 0 = bearish energy
        bullish_energy = bull_power[i] > 0
        bearish_energy = bear_power[i] < 0
        
        # ADX filter: trending market only
        trending = adx_1w_aligned[i] > 25
        
        if position == 0:
            # Enter long: Alligator aligned up + Bull Power positive + trending
            if alligator_long and bullish_energy and trending:
                position = 1
                signals[i] = position_size
            # Enter short: Alligator aligned down + Bear Power negative + trending
            elif alligator_short and bearish_energy and trending:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Alligator alignment breaks OR Bear Power becomes negative
            if not (alligator_long and bullish_energy):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Alligator alignment breaks OR Bull Power becomes positive
            if not (alligator_short and bearish_energy):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1w_alligator_elder_ray_adx_v1"
timeframe = "12h"
leverage = 1.0