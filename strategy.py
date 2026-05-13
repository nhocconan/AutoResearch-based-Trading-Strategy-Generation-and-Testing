#!/usr/bin/env python3
"""
6h_1d_ChaikinOscillator_Trend
Hypothesis: Chaikin Oscillator (3,10) on 1d provides institutional accumulation/distribution signals. 
Trade in direction of 1d trend (EMA50) when Chaikin Oscillator crosses zero with confirmation from 
increasing volume. Works in both bull/bear markets by following higher timeframe trend. 
Target: 15-25 trades/year per symbol.
"""

name = "6h_1d_ChaikinOscillator_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Chaikin Oscillator and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
    # Avoid division by zero
    hl_range = high_1d - low_1d
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    mfm = ((close_1d - low_1d) - (high_1d - close_1d)) / hl_range
    
    # Money Flow Volume = MFM * Volume
    mfv = mfm * volume_1d
    
    # Chaikin Oscillator = (3-period EMA of MFV) - (10-period EMA of MFV)
    mfv_series = pd.Series(mfv)
    ema3 = mfv_series.ewm(span=3, adjust=False, min_periods=3).mean().values
    ema10 = mfv_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    chaikin_osc = ema3 - ema10
    
    # 1d trend: 50 EMA
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = close_1d > ema_50_1d
    downtrend_1d = close_1d < ema_50_1d
    
    # Align 1d indicators to 6h
    chaikin_osc_aligned = align_htf_to_ltf(prices, df_1d, chaikin_osc)
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average volume on 6h
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.5)
    
    # Detect Chaikin Oscillator zero crosses
    # Bullish cross: CO crosses above zero
    bullish_cross = (chaikin_osc_aligned > 0) & (np.roll(chaikin_osc_aligned, 1) <= 0)
    # Bearish cross: CO crosses below zero
    bearish_cross = (chaikin_osc_aligned < 0) & (np.roll(chaikin_osc_aligned, 1) >= 0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Handle first element roll
        if i == 0:
            bullish_cross[i] = False
            bearish_cross[i] = False
        
        # Get aligned values
        uptrend = uptrend_1d_aligned[i]
        downtrend = downtrend_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # LONG: 1d uptrend + bullish Chaikin cross + volume spike
            if uptrend and bullish_cross[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: 1d downtrend + bearish Chaikin cross + volume spike
            elif downtrend and bearish_cross[i] and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: 1d trend turns down or bearish Chaikin cross
            if not uptrend or bearish_cross[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: 1d trend turns up or bullish Chaikin cross
            if not downtrend or bullish_cross[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals