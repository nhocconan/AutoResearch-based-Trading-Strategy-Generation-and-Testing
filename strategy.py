#!/usr/bin/env python3
# 1d_Williams_Alligator_ElderRay_Trend_Follower
# Hypothesis: Combine Williams Alligator trend detection with Elder Ray power to capture strong trends across bull and bear markets.
# Uses Alligator (Jaw/Teeth/Lips) for trend direction and Elder Ray (Bull/Bear Power) for momentum confirmation.
# Filters by weekly ADX > 25 to ensure trending markets. Designed for low-frequency, high-conviction trades on daily timeframe.

name = "1d_Williams_Alligator_ElderRay_Trend_Follower"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(series, period):
    """Smoothed Moving Average (SMMA) - used in Williams Alligator"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=float)
    result = np.full_like(series, np.nan, dtype=float)
    # First value is simple moving average
    result[period-1] = np.mean(series[:period])
    # Subsequent values: SMMA = (prev_smma * (period-1) + current_value) / period
    for i in range(period, len(series)):
        result[i] = (result[i-1] * (period-1) + series[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for ADX filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly ADX (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smooth TR, DM+ , DM- (14-period)
    tr14 = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / (tr14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (tr14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False).mean().values
    adx_14_1w = adx  # Already smoothed
    
    # Align weekly ADX to daily
    adx_14_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_14_1w)

    # Williams Alligator (using SMMA)
    # Jaw: 13-period SMMA, shifted 8 bars
    jaw = smma(close, 13)
    jaw = np.roll(jaw, 8)  # Shift forward 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars
    teeth = smma(close, 8)
    teeth = np.roll(teeth, 5)  # Shift forward 5 bars
    # Lips: 5-period SMMA, shifted 3 bars
    lips = smma(close, 5)
    lips = np.roll(lips, 3)  # Shift forward 3 bars

    # Elder Ray: Bull Power and Bear Power (13-period EMA)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup
        # Skip if any required value is NaN
        if (np.isnan(adx_14_1w_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_long = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_short = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Elder Ray confirmation: Bull Power > 0 and rising, Bear Power < 0 and falling
        # Simplified: Bull Power > 0 for long, Bear Power < 0 for short
        elder_long = bull_power[i] > 0
        elder_short = bear_power[i] < 0
        
        # Weekly trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_14_1w_aligned[i] > 25

        if position == 0:
            # LONG: Alligator aligned up + Bull Power positive + strong weekly trend
            if alligator_long and elder_long and strong_trend:
                signals[i] = 0.25
                position = 1
            # SHORT: Alligator aligned down + Bear Power negative + strong weekly trend
            elif alligator_short and elder_short and strong_trend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator alignment breaks or Bull Power turns negative
            if not (alligator_long and elder_long):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator alignment breaks or Bear Power turns positive
            if not (alligator_short and elder_short):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals