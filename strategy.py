#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Alligator_ElderRay_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for Elder Ray and Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = df_1d['high'].values - ema13
    bear_power = df_1d['low'].values - ema13
    
    # Alligator: Jaw (13), Teeth (8), Lips (5) - all SMAs
    jaw = pd.Series(df_1d['close']).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(df_1d['close']).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).mean().values
    
    # Align all indicators to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(13, n):
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bull = bull_power_aligned[i]
        bear = bear_power_aligned[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        
        if position == 0:
            # Enter long: Bull Power > 0, Bear Power < 0, and Lips > Teeth > Jaw (bullish alignment)
            if bull > 0 and bear < 0 and lips_val > teeth_val > jaw_val:
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power > 0, Bull Power < 0, and Lips < Teeth < Jaw (bearish alignment)
            elif bear > 0 and bull < 0 and lips_val < teeth_val < jaw_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bull Power turns negative or Bear Power turns positive
            if bull <= 0 or bear >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power turns negative or Bull Power turns positive
            if bear <= 0 or bull >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals