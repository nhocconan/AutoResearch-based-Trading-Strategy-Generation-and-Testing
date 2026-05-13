#!/usr/bin/env python3
"""
4h_1d_ChaikinOscillator_Trend
Hypothesis: Chaikin Oscillator (3,10) on 1d combined with 1d EMA(50) trend filter. 
Chaikin Oscillator > 0 indicates buying pressure, < 0 selling pressure. 
Follow the higher timeframe trend: long when 1d trend up and Chaikin > 0, short when 1d trend down and Chaikin < 0.
Uses volume-weighted accumulation/distribution to filter noise, effective in both bull and bear markets.
Target: 20-40 trades/year per symbol.
"""

name = "4h_1d_ChaikinOscillator_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Chaikin Oscillator on 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Money Flow Multiplier
    mfm = np.where((high_1d - low_1d) != 0, 
                   ((close_1d - low_1d) - (high_1d - close_1d)) / (high_1d - low_1d), 
                   0)
    # Money Flow Volume
    mfv = mfm * volume_1d
    
    # Accumulation/Distribution Line
    adl = np.cumsum(mfv)
    
    # Chaikin Oscillator = EMA(3, ADL) - EMA(10, ADL)
    adl_series = pd.Series(adl)
    ema3 = adl_series.ewm(span=3, adjust=False, min_periods=3).mean().values
    ema10 = adl_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    chaikin = ema3 - ema10
    
    # 1d trend: EMA(50)
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = close_1d > ema50_1d
    downtrend_1d = close_1d < ema50_1d
    
    # Align 1d indicators to 4h
    chaikin_aligned = align_htf_to_ltf(prices, df_1d, chaikin)
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get aligned values
        chaikin_val = chaikin_aligned[i]
        uptrend = uptrend_1d_aligned[i]
        downtrend = downtrend_1d_aligned[i]
        
        if position == 0:
            # LONG: 1d uptrend + positive Chaikin (buying pressure)
            if uptrend and chaikin_val > 0:
                signals[i] = 0.25
                position = 1
            # SHORT: 1d downtrend + negative Chaikin (selling pressure)
            elif downtrend and chaikin_val < 0:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: 1d trend turns down or Chaikin turns negative
            if not uptrend or chaikin_val <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: 1d trend turns up or Chaikin turns positive
            if not downtrend or chaikin_val >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals