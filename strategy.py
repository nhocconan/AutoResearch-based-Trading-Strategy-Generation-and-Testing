#!/usr/bin/env python3
"""
6h_12h_1d_Alligator_ElderRay_v1
Hypothesis: Combine Williams Alligator (trend filter) and Elder Ray (bull/bear power) on higher timeframes.
Use 12h Alligator (jaw/teeth/lips) to define trend direction, and 1d Elder Ray to measure bull/bear power.
Go long when price > Alligator teeth AND bull power > 0; short when price < Alligator teeth AND bear power < 0.
Exit when price crosses Alligator jaw (trend change) or power weakens.
Works in bull/bear because Alligator adapts to price action and Elder Ray measures underlying strength.
Target: 20-40 trades/year per symbol.
"""

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
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === 12h Data (for Alligator) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams Alligator (13,8,5 SMAs shifted forward)
    # Jaw (13-period SMMA, shifted 8 bars)
    jaw_12h = pd.Series(close_12h).rolling(window=13, min_periods=13).mean()
    jaw_12h = jaw_12h.shift(8)  # future shift
    # Teeth (8-period SMMA, shifted 5 bars)
    teeth_12h = pd.Series(close_12h).rolling(window=8, min_periods=8).mean()
    teeth_12h = teeth_12h.shift(5)  # future shift
    # Lips (5-period SMMA, shifted 3 bars)
    lips_12h = pd.Series(close_12h).rolling(window=5, min_periods=5).mean()
    lips_12h = lips_12h.shift(3)  # future shift
    
    # Align to 6h timeframe (wait for 12h bar close)
    jaw_12h_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h.values)
    teeth_12h_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h.values)
    lips_12h_aligned = align_htf_to_ltf(prices, df_12h, lips_12h.values)
    
    # === 1d Data (for Elder Ray) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean()
    bull_power_1d = high_1d - ema13_1d.values
    bear_power_1d = low_1d - ema13_1d.values
    
    # Align to 6h timeframe (wait for 1d bar close)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(jaw_12h_aligned[i]) or 
            np.isnan(teeth_12h_aligned[i]) or
            np.isnan(lips_12h_aligned[i]) or
            np.isnan(bull_power_1d_aligned[i]) or
            np.isnan(bear_power_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price > Alligator teeth AND bull power > 0
            if close[i] > teeth_12h_aligned[i] and bull_power_1d_aligned[i] > 0:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price < Alligator teeth AND bear power < 0
            elif close[i] < teeth_12h_aligned[i] and bear_power_1d_aligned[i] < 0:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: price crosses Alligator jaw (trend change) or power weakens
        elif position == 1:
            # Exit long: price crosses below jaw OR bull power <= 0
            if close[i] <= jaw_12h_aligned[i] or bull_power_1d_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above jaw OR bear power >= 0
            if close[i] >= jaw_12h_aligned[i] or bear_power_1d_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_12h_1d_Alligator_ElderRay_v1"
timeframe = "6h"
leverage = 1.0