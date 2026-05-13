#!/usr/bin/env python3
"""
6h_ADX_Alligator_BullBear_Power
Hypothesis: Combines ADX for trend strength, Williams Alligator for directional bias, and Elder Ray's Bull/Bear Power for momentum confirmation. Works in both bull and bear markets by using Alligator jaws-teeth-lips alignment to determine trend direction, ADX to filter weak trends, and Bull/Bear Power to confirm momentum behind the move. Designed for low trade frequency (~20-40/year) to minimize fee drag.
"""

name = "6h_ADX_Alligator_BullBear_Power"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Alligator and ADX (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Williams Alligator on 1d: Jaw (13), Teeth (8), Lips (5) SMMA
    # SMMA (Smoothed Moving Average) approximation using EMA with alpha = 1/period
    close_1d = df_1d['close'].values
    jaw_period, teeth_period, lips_period = 13, 8, 5
    alpha_jaw = 1.0 / jaw_period
    alpha_teeth = 1.0 / teeth_period
    alpha_lips = 1.0 / lips_period
    
    # Initialize SMMA arrays
    jaw = np.full_like(close_1d, np.nan)
    teeth = np.full_like(close_1d, np.nan)
    lips = np.full_like(close_1d, np.nan)
    
    # First value is simple average
    jaw[jaw_period-1] = np.mean(close_1d[:jaw_period])
    teeth[teeth_period-1] = np.mean(close_1d[:teeth_period])
    lips[lips_period-1] = np.mean(close_1d[:lips_period])
    
    # Subsequent values: SMMA = (prev_smma * (period-1) + close) / period
    for i in range(jaw_period, len(close_1d)):
        jaw[i] = (jaw[i-1] * (jaw_period-1) + close_1d[i]) / jaw_period
    for i in range(teeth_period, len(close_1d)):
        teeth[i] = (teeth[i-1] * (teeth_period-1) + close_1d[i]) / teeth_period
    for i in range(lips_period, len(close_1d)):
        lips[i] = (lips[i-1] * (lips_period-1) + close_1d[i]) / lips_period
    
    # Alligator alignment: Bullish when Lips > Teeth > Jaw, Bearish when Lips < Teeth < Jaw
    alligator_bullish = (lips > teeth) & (teeth > jaw)
    alligator_bearish = (lips < teeth) & (teeth < jaw)
    
    # ADX on 1d to measure trend strength
    # Calculate True Range
    tr1 = df_1d['high'][1:].values - df_1d['low'][1:].values
    tr2 = np.abs(df_1d['high'][1:].values - df_1d['close'][:-1].values)
    tr3 = np.abs(df_1d['low'][1:].values - df_1d['close'][:-1].values)
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = df_1d['high'][1:].values - df_1d['high'][:-1].values
    down_move = df_1d['low'][:-1].values - df_1d['low'][1:].values
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    adx_period = 14
    alpha_adx = 1.0 / adx_period
    
    atr = np.full_like(tr, np.nan)
    plus_di = np.full_like(tr, np.nan)
    minus_di = np.full_like(tr, np.nan)
    
    # First ATR is average of first 'adx_period' TR values
    atr[adx_period] = np.nanmean(tr[1:adx_period+1])
    plus_dm_smooth = np.full_like(plus_dm, np.nan)
    minus_dm_smooth = np.full_like(minus_dm, np.nan)
    plus_dm_smooth[adx_period] = np.nansum(plus_dm[1:adx_period+1])
    minus_dm_smooth[adx_period] = np.nansum(minus_dm[1:adx_period+1])
    
    # Subsequent values using Wilder's smoothing
    for i in range(adx_period+1, len(tr)):
        atr[i] = (atr[i-1] * (adx_period-1) + tr[i]) / adx_period
        plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (adx_period-1) + plus_dm[i]) / adx_period
        minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (adx_period-1) + minus_dm[i]) / adx_period
    
    # Avoid division by zero
    atr_safe = np.where(atr == 0, 1e-10, atr)
    plus_di = 100 * plus_dm_smooth / atr_safe
    minus_di = 100 * minus_dm_smooth / atr_safe
    
    # DX and ADX
    dx = np.full_like(tr, np.nan)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    
    adx = np.full_like(tr, np.nan)
    # First ADX is average of first 'adx_period' DX values
    adx[2*adx_period-1] = np.nanmean(dx[adx_period:2*adx_period])
    for i in range(2*adx_period, len(dx)):
        adx[i] = (adx[i-1] * (adx_period-1) + dx[i]) / adx_period
    
    # Align Alligator and ADX to 6t timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    bullish_aligned = align_htf_to_ltf(prices, df_1d, alligator_bullish.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_1d, alligator_bearish.astype(float))
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Elder Ray Bull/Bear Power on 6t data
    # Calculate EMA(13) for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA
    bear_power = low - ema13   # Bear Power = Low - EMA
    
    # Generate signals
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Warmup period for indicators
        if position == 0:
            # LONG: Alligator bullish, ADX > 25 (strong trend), Bull Power > 0 and rising
            if (bullish_aligned[i] and 
                adx_aligned[i] > 25 and 
                bull_power[i] > 0 and 
                (i == 30 or bull_power[i] > bull_power[i-1])):
                signals[i] = 0.25
                position = 1
            # SHORT: Alligator bearish, ADX > 25 (strong trend), Bear Power < 0 and falling
            elif (bearish_aligned[i] and 
                  adx_aligned[i] > 25 and 
                  bear_power[i] < 0 and 
                  (i == 30 or bear_power[i] < bear_power[i-1])):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator turns bearish OR ADX weakens (< 20) OR Bull Power turns negative
            if (not bullish_aligned[i] or 
                adx_aligned[i] < 20 or 
                bull_power[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator turns bullish OR ADX weakens (< 20) OR Bear Power turns positive
            if (not bearish_aligned[i] or 
                adx_aligned[i] < 20 or 
                bear_power[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals