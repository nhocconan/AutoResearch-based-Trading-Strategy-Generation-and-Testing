#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d Elder Ray volume confirmation.
# Long when Alligator jaws (13-period SMMA) > teeth (8-period SMMA) > lips (5-period SMMA) and 1d Bull Power > 0.
# Short when Alligator jaws < teeth < lips and 1d Bear Power < 0.
# Uses discrete position sizing (0.25) to minimize fee drag.
# Target: 80-160 total trades over 4 years (20-40/year) to avoid excessive fee churn.
# Williams Alligator identifies trend alignment; Elder Ray confirms bull/bear power from higher timeframe.

name = "4h_Williams_Alligator_1dElderRay_VolumeConfirm_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate 1d Elder Ray (Bull Power and Bear Power)
    # Bull Power = High - EMA13(Close)
    # Bear Power = Low - EMA13(Close)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema_13_1d
    bear_power = low_1d - ema_13_1d
    
    # Align Elder Ray to 4h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Williams Alligator: SMMA (Smoothed Moving Average) - approximates with EMA for simplicity
    # Jaws: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # Using EMA as proxy for SMMA (common approximation)
    jaws = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 13  # Ensure sufficient history for Alligator
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Alligator conditions: jaws > teeth > lips (bullish) OR jaws < teeth < lips (bearish)
        alligator_bullish = jaws[i] > teeth[i] and teeth[i] > lips[i]
        alligator_bearish = jaws[i] < teeth[i] and teeth[i] < lips[i]
        
        # Elder Ray conditions: Bull Power > 0 (bullish) OR Bear Power < 0 (bearish)
        elder_bullish = bull_power_aligned[i] > 0
        elder_bearish = bear_power_aligned[i] < 0
        
        # Entry conditions
        if alligator_bullish and elder_bullish and position <= 0:
            signals[i] = 0.25
            position = 1
        elif alligator_bearish and elder_bearish and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: Alligator reverses or Elder Ray diverges
        elif position == 1 and (not alligator_bullish or not elder_bullish):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not alligator_bearish or not elder_bearish):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals