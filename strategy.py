#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator + 1d Trend Filter + Volume Spike
# Enters long when price > Alligator Jaw (13-period SMMA) with 1d EMA50 uptrend and volume > 1.8x MA20.
# Enters short when price < Alligator Jaw with 1d EMA50 downtrend and volume > 1.8x MA20.
# Exits when price crosses the Alligator Teeth (8-period SMMA).
# Uses discrete sizing (0.25) to limit fee drawdown. Target: 12-25 trades/year on 12h.
# Williams Alligator uses SMMA (smoothed moving average) which is less whipsaw-prone in ranging/ bear markets.
# 1d trend filter ensures alignment with higher-timeframe momentum, reducing false breakouts.
# Volume spike confirms institutional participation. Designed for low frequency and high conviction.

name = "12h_WilliamsAlligator_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) aka Wilder's smoothing"""
    if length < 1:
        return np.full_like(source, np.nan, dtype=np.float64)
    result = np.full_like(source, np.nan, dtype=np.float64)
    if len(source) < length:
        return result
    # First value is simple average
    result[length-1] = np.mean(source[:length])
    # Subsequent values: SMMA = (PREV_SMMA * (length-1) + CURRENT) / length
    for i in range(length, len(source)):
        result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams Alligator (Jaw, Teeth, Lips)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) - all SMMA
    jaw_12h = smma(df_12h['close'].values, 13)
    teeth_12h = smma(df_12h['close'].values, 8)
    lips_12h = smma(df_12h['close'].values, 5)
    
    jaw_12h_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_12h_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_12h_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 1.8x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data for all indicators
        if np.isnan(jaw_12h_aligned[i]) or np.isnan(teeth_12h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price > Jaw, 1d uptrend, volume spike
            if close[i] > jaw_12h_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price < Jaw, 1d downtrend, volume spike
            elif close[i] < jaw_12h_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Teeth (weaker trend signal)
            if close[i] < teeth_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above Teeth
            if close[i] > teeth_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals