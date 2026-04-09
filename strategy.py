#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams Alligator + Elder Ray + volume confirmation
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend direction and strength
# Elder Ray (Bull/Bear Power = EMA13) measures trend momentum
# Volume confirmation filters weak breakouts (current 4h volume > 1.5x 20-period average)
# ATR trailing stop (2.5x ATR) manages risk and reduces whipsaw
# Designed for 4h timeframe targeting 20-50 trades/year (80-200 over 4 years)
# Works in bull/bear: Alligator alignment shows trend, Elder Ray confirms strength, volume validates, ATR stop adapts

name = "4h_1d_alligator_elder_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams Alligator
    # Jaw (Blue): 13-period SMMA, shifted 8 bars
    sma_13 = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(sma_13, 8)
    # Teeth (Red): 8-period SMMA, shifted 5 bars
    sma_8 = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(sma_8, 5)
    # Lips (Green): 5-period SMMA, shifted 3 bars
    sma_5 = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(sma_5, 3)
    
    # Align 1d Alligator to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate 1d Elder Ray (EMA13 for Bull/Bear Power)
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema_13  # Bull Power = High - EMA13
    bear_power = low_1d - ema_13   # Bear Power = Low - EMA13
    
    # Align 1d Elder Ray to 4h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Pre-compute ATR(14) for 4h timeframe
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 4h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Alligator trend detection: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_long = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        alligator_short = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        # Elder Ray confirmation: Bull Power > 0 and rising, Bear Power < 0 and falling
        # Simplified: Bull Power > 0 for long, Bear Power < 0 for short
        elder_long = bull_power_aligned[i] > 0
        elder_short = bear_power_aligned[i] < 0
        
        if position == 1:  # Long position
            # Update highest high since entry
            if close[i] > highest_since_long:
                highest_since_long = close[i]
            # ATR trailing stop: exit if price drops 2.5x ATR from highest
            if close[i] < highest_since_long - 2.5 * atr[i]:
                position = 0
                highest_since_long = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if close[i] < lowest_since_short:
                lowest_since_short = close[i]
            # ATR trailing stop: exit if price rises 2.5x ATR from lowest
            if close[i] > lowest_since_short + 2.5 * atr[i]:
                position = 0
                lowest_since_short = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter on Alligator alignment + Elder Ray confirmation + volume
            if alligator_long and elder_long and volume_confirmed:
                position = 1
                highest_since_long = close[i]
                signals[i] = 0.25
            elif alligator_short and elder_short and volume_confirmed:
                position = -1
                lowest_since_short = close[i]
                signals[i] = -0.25
    
    return signals