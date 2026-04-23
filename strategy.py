#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator + Elder Ray (Bull/Bear Power) + volume confirmation.
Long when: Alligator jaws < teeth < lips (bullish alignment) AND Bull Power > 0 AND volume > 1.5x 20-period average.
Short when: Alligator jaws > teeth > lips (bearish alignment) AND Bear Power < 0 AND volume > 1.5x 20-period average.
Exit when Alligator alignment reverses or volume drops below average.
Uses 1d EMA13 for Alligator and Elder Ray calculations. Targets 12-25 trades/year per symbol.
Alligator filters choppy markets, Elder Ray measures trend strength, volume confirms conviction.
"""

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
    
    # Load 1d data for Alligator and Elder Ray - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Alligator lines (Smoothed Moving Average - using EMA as approximation)
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    jaw_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth_1d = pd.Series(close_1d).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips_1d = pd.Series(close_1d).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13_1d = jaw_1d  # Jaw is EMA13
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # Volume average (20-period) on 1d timeframe
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ma_val = vol_ma_aligned[i]
        
        if position == 0:
            # Long: Bullish Alligator alignment AND Bull Power > 0 AND volume confirmation
            if (jaw_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < lips_aligned[i] and
                bull_power_aligned[i] > 0 and volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment AND Bear Power < 0 AND volume confirmation
            elif (jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i] and
                  bear_power_aligned[i] < 0 and volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Alligator alignment turns bearish OR volume drops below average
                if not (jaw_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < lips_aligned[i]):
                    exit_signal = True
                elif volume[i] < vol_ma_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Alligator alignment turns bullish OR volume drops below average
                if not (jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i]):
                    exit_signal = True
                elif volume[i] < vol_ma_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Williams_Alligator_ElderRay_VolumeConfirm"
timeframe = "6h"
leverage = 1.0