#!/usr/bin/env python3
"""
4h_Chaikin_Oscillator_Pullback_1dTrend
Hypothesis: Chaikin Oscillator (3,10) zero-cross pullbacks with 1-day EMA50 trend filter and volume spike.
Works in both bull/bear markets by combining momentum (Chaikin) with trend (EMA50) and volume confirmation.
Target: 20-30 trades/year to minimize fee drag while capturing swings.
"""

name = "4h_Chaikin_Oscillator_Pullback_1dTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Chaikin Oscillator (3,10) on price data
    # Chaikin Oscillator = EMA(3, ADL) - EMA(10, ADL)
    # ADL = Cumsum( ((Close - Low) - (High - Close)) / (High - Low) * Volume )
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Money Flow Multiplier
    mfm = ((close - low) - (high - close)) / (high - low)
    # Replace division by zero or NaN with 0
    mfm = np.where((high - low) == 0, 0, mfm)
    mfm = np.where(np.isnan(mfm), 0, mfm)
    
    # Money Flow Volume
    mfv = mfm * volume
    
    # Accumulation/Distribution Line
    adl = np.cumsum(mfv)
    
    # Chaikin Oscillator: EMA(3, ADL) - EMA(10, ADL)
    adl_series = pd.Series(adl)
    chaikin_osc = (
        adl_series.ewm(span=3, adjust=False, min_periods=3).mean().values -
        adl_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    )
    
    # Volume filter: current volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50 (50) and Chaikin Oscillator (10)
    start_idx = max(50, 10)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(chaikin_osc[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Chaikin crosses above zero with uptrend and volume
            if chaikin_osc[i] > 0 and chaikin_osc[i-1] <= 0 and close[i] > ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Chaikin crosses below zero with downtrend and volume
            elif chaikin_osc[i] < 0 and chaikin_osc[i-1] >= 0 and close[i] < ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Chaikin crosses below zero or trend change
            if chaikin_osc[i] < 0 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Chaikin crosses above zero or trend change
            if chaikin_osc[i] > 0 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals