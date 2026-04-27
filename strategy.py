#!/usr/bin/env python3
"""
#100844 - 1d_ChaikinOscillator_Breakout_1wTrend_Volume
Hypothesis: Daily Chaikin Oscillator (3,10) crossing zero with weekly trend filter and volume confirmation.
Works in bull (breakouts with trend) and bear (mean reversion via Chaikin reversals). Targets 10-25 trades/year.
Uses 1d primary timeframe with 1w HTF for trend filter.
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA20 for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate Chaikin Oscillator (3,10) on daily data
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    # Money Flow Volume = Money Flow Multiplier * Volume
    # ADL = cumulative sum of Money Flow Volume
    # Chaikin Oscillator = EMA(3, ADL) - EMA(10, ADL)
    
    # Avoid division by zero
    hl_range = high - low
    hl_range = np.where(hl_range == 0, 1, hl_range)
    
    mfm = ((close - low) - (high - close)) / hl_range
    mfv = mfm * volume
    
    # Calculate ADL (Accumulation/Distribution Line)
    adl = np.cumsum(mfv)
    
    # Calculate EMAs of ADL
    adl_series = pd.Series(adl)
    ema3_adl = adl_series.ewm(span=3, adjust=False, min_periods=3).mean().values
    ema10_adl = adl_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Chaikin Oscillator
    chaikin_osc = ema3_adl - ema10_adl
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(chaikin_osc[i]) or 
            i < 10):  # Need enough data for Chaikin calculation
            signals[i] = 0.0
            continue
        
        # Volume filter: volume > 1.3x 20-period average
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            volume_filter = volume[i] > (vol_ma * 1.3)
        else:
            volume_filter = False
        
        # Long condition: Chaikin crosses above zero, above weekly EMA20, volume
        if (chaikin_osc[i] > 0 and chaikin_osc[i-1] <= 0 and 
            close[i] > ema20_1w_aligned[i] and 
            volume_filter):
            signals[i] = 0.25
            position = 1
        # Short condition: Chaikin crosses below zero, below weekly EMA20, volume
        elif (chaikin_osc[i] < 0 and chaikin_osc[i-1] >= 0 and 
              close[i] < ema20_1w_aligned[i] and 
              volume_filter):
            signals[i] = -0.25
            position = -1
        # Exit conditions: Chaikin crosses zero in opposite direction
        elif position == 1 and chaikin_osc[i] < 0 and chaikin_osc[i-1] >= 0:
            signals[i] = 0.0
            position = 0
        elif position == -1 and chaikin_osc[i] > 0 and chaikin_osc[i-1] <= 0:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_ChaikinOscillator_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0