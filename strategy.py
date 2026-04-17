#!/usr/bin/env python3
"""
4h_Chaikin_Oscillator_Plus_Volume
Hypothesis: Chaikin Oscillator (3,10) combined with volume confirmation and ATR-based trend filter
works in both bull and bear markets by capturing institutional money flow with volume validation.
Designed for low trade frequency (15-30/year) on 4h timeframe to minimize fee drag.
"""

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
    
    # === Chaikin Oscillator components (daily) ===
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    # Avoid division by zero
    hl_range = high - low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    mfm = ((close - low) - (high - close)) / hl_range
    # Money Flow Volume = Money Flow Multiplier * Volume
    mfv = mfm * volume
    
    # Accumulation/Distribution Line
    adl = np.cumsum(mfv)
    
    # Chaikin Oscillator = EMA(3, ADL) - EMA(10, ADL)
    adl_series = pd.Series(adl)
    ema3 = adl_series.ewm(span=3, adjust=False, min_periods=3).mean().values
    ema10 = adl_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    chaikin = ema3 - ema10
    
    # === Volume confirmation (daily average) ===
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Trend filter: 4h EMA34 (avoid whipsaw) ===
    close_series = pd.Series(close)
    ema34 = close_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    signals = np.zeros(n)
    
    # Warmup period: enough for EMA10 and EMA34
    warmup = 40
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(chaikin[i]) or np.isnan(vol_avg_20[i]) or 
            np.isnan(ema34[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume filter: current volume > 1.3x 20-period average
        vol_filter = volume[i] > 1.3 * vol_avg_20[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Chaikin crosses above zero + volume filter + price above EMA34
            if chaikin[i] > 0 and chaikin[i-1] <= 0 and vol_filter and close[i] > ema34[i]:
                signals[i] = 0.25
                position = 1
                continue
            # Short: Chaikin crosses below zero + volume filter + price below EMA34
            elif chaikin[i] < 0 and chaiken[i-1] >= 0 and vol_filter and close[i] < ema34[i]:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse signal or trend deterioration
        elif position == 1:
            # Exit when Chaikin crosses below zero OR price falls below EMA34
            if chaikin[i] < 0 and chaikin[i-1] >= 0 or close[i] < ema34[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when Chaikin crosses above zero OR price rises above EMA34
            if chaikin[i] > 0 and chaikin[i-1] <= 0 or close[i] > ema34[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Chaikin_Oscillator_Plus_Volume"
timeframe = "4h"
leverage = 1.0