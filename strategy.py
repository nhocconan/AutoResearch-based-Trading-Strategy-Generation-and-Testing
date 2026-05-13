#!/usr/bin/env python3
"""
12h_Chaikin_Oscillator_Breakout_1dTrend_Volume
Hypothesis: Use daily Chaikin Oscillator (3,10) zero-line cross to detect accumulation/distribution shifts, confirmed by price breaking above/below 12h Donchian channel (20-period) and volume spike (>1.5x 20-period average). Works in bull markets by catching early accumulation and in bear markets by detecting distribution. Designed for 12h timeframe to limit trades (target 50-150 total over 4 years) and avoid fee drag.
"""

name = "12h_Chaikin_Oscillator_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Chaikin Oscillator calculation
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
    
    # Calculate Chaikin Oscillator: (3-day EMA of ADL) - (10-day EMA of ADL)
    adl_series = pd.Series(adl)
    ema3 = adl_series.ewm(span=3, adjust=False, min_periods=3).mean().values
    ema10 = adl_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    chaikin_osc = ema3 - ema10
    
    # Align Chaikin Oscillator to 12h timeframe (no extra delay needed)
    chaikin_osc_aligned = align_htf_to_ltf(prices, df_1d, chaikin_osc)
    
    # Get 12h Donchian channel (20-period) for breakout signals
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume average (20-period) for volume spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(chaikin_osc_aligned[i]) or np.isnan(chaikin_osc_aligned[i-1]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5x 20-period average
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Chaikin Oscillator crosses above zero (accumulation) + price breaks above Donchian high + volume spike
            if chaikin_osc_aligned[i-1] <= 0 and chaikin_osc_aligned[i] > 0 and close[i] > donchian_high[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: Chaikin Oscillator crosses below zero (distribution) + price breaks below Donchian low + volume spike
            elif chaikin_osc_aligned[i-1] >= 0 and chaikin_osc_aligned[i] < 0 and close[i] < donchian_low[i] and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Chaikin Oscillator crosses below zero or price breaks below Donchian low
            if chaikin_osc_aligned[i] < 0 or close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Chaikin Oscillator crosses above zero or price breaks above Donchian high
            if chaikin_osc_aligned[i] > 0 or close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals