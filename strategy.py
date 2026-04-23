#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using 1d Williams Alligator + Elder Ray + volume confirmation.
Williams Alligator (Jaw/Teeth/Lips) defines trend direction and alignment.
Elder Ray (Bull/Bear Power) measures trend strength via EMA13.
Long when: Alligator aligned bullish (Lips>Teeth>Jaw) AND Bull Power > 0 AND volume > 1.5x 20-period average.
Short when: Alligator aligned bearish (Lips<Teeth<Jaw) AND Bear Power < 0 AND volume > 1.5x 20-period average.
Exit when: Alligator alignment breaks OR volume drops below average.
Uses discrete position sizing (0.25) to control drawdown and fee churn.
Designed for 6h timeframe to target 12-37 trades/year per symbol (50-150 total over 4 years).
Williams Alligator filters whipsaws in sideways markets; Elder Ray confirms trend strength.
Works in both bull and bear markets by requiring volume confirmation and strict alignment.
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
    
    # Calculate 1d Williams Alligator and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator: SMAs with specific offsets
    # Jaw: 13-period SMA, offset 8 bars
    # Teeth: 8-period SMA, offset 5 bars  
    # Lips: 5-period SMA, offset 3 bars
    jaw_1d = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth_1d = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    lips_1d = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align Alligator lines and Elder Ray to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 13+8, 8+5, 5+3)  # max of Alligator offsets + vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        
        if position == 0:
            # Long: Alligator bullish alignment AND Bull Power positive AND volume spike
            if (lips_val > teeth_val and teeth_val > jaw_val and 
                bull_power_val > 0 and volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish alignment AND Bear Power negative AND volume spike
            elif (lips_val < teeth_val and teeth_val < jaw_val and 
                  bear_power_val < 0 and volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Update position holding
            signals[i] = 0.25 if position == 1 else -0.25
            
            # Exit conditions
            exit_signal = False
            
            # Exit: Alligator alignment breaks OR volume drops below average
            if position == 1:
                if not (lips_val > teeth_val and teeth_val > jaw_val):
                    exit_signal = True
                elif volume[i] < vol_ma_val:  # volume drop
                    exit_signal = True
            elif position == -1:
                if not (lips_val < teeth_val and teeth_val < jaw_val):
                    exit_signal = True
                elif volume[i] < vol_ma_val:  # volume drop
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6H_WilliamsAlligator_ElderRay_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0