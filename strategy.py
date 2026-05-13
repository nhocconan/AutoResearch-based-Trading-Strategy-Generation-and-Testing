#!/usr/bin/env python3
"""
12h_1d_ChaikinOscillator_Trend_Momentum
Hypothesis: Chaikin Oscillator (3,10) on 1d chart measures accumulation/distribution momentum.
When Chaikin Oscillator crosses above zero with 1d uptrend and volume confirmation,
it signals bullish momentum for long entries. When it crosses below zero with 1d downtrend,
it signals bearish momentum for short entries. Uses 12h timeframe for execution.
Works in both bull and bear markets by following 1d momentum and trend.
Target: 12-37 trades/year per symbol.
"""

name = "12h_1d_ChaikinOscillator_Trend_Momentum"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Chaikin Oscillator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Money Flow Multiplier and Money Flow Volume
    mfm = ((close_1d - low_1d) - (high_1d - close_1d)) / (high_1d - low_1d)
    mfm = np.where((high_1d - low_1d) == 0, 0, mfm)  # avoid division by zero
    mfv = mfm * volume_1d
    
    # Calculate Accumulation/Diffusion Line (ADL)
    adl = np.cumsum(mfv)
    
    # Chaikin Oscillator = EMA(3, ADL) - EMA(10, ADL)
    adl_series = pd.Series(adl)
    ema3 = adl_series.ewm(span=3, adjust=False, min_periods=3).mean().values
    ema10 = adl_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    chaikin_osc = ema3 - ema10
    
    # Align Chaikin Oscillator to 12h timeframe (wait for 1d bar to close)
    chaikin_osc_aligned = align_htf_to_ltf(prices, df_1d, chaikin_osc)
    
    # 1d trend: 34 EMA
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    uptrend_1d = close_1d > ema_34_1d
    downtrend_1d = close_1d < ema_34_1d
    
    # Align 1d trend to 12h
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get aligned values for current bar
        chaikin = chaikin_osc_aligned[i]
        uptrend = uptrend_1d_aligned[i]
        downtrend = downtrend_1d_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: Chaikin Oscillator crosses above zero, 1d uptrend, volume confirmation
            if chaikin > 0 and uptrend and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: Chaikin Oscillator crosses below zero, 1d downtrend, volume confirmation
            elif chaikin < 0 and downtrend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Chaikin Oscillator crosses below zero or 1d trend turns down
            if chaikin < 0 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Chaikin Oscillator crosses above zero or 1d trend turns up
            if chaikin > 0 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals