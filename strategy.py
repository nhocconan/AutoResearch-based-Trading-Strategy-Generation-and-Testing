#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray + volume confirmation
# - Williams Alligator: Jaw (13-period SMMA, 8 offset), Teeth (8-period SMMA, 5 offset), Lips (5-period SMMA, 3 offset)
#   Long when Lips > Teeth > Jaw (bullish alignment), Short when Lips < Teeth < Jaw (bearish alignment)
# - Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
#   Long when Bull Power > 0 and rising, Short when Bear Power > 0 and rising
# - Volume confirmation: Current volume > 1.5x 20-period average
# - Uses 1w for Alligator/Elder Ray calculation (strong trend filter) and 12h for execution
# - Target: 15-30 trades per year per symbol (60-120 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data for Williams Alligator and Elder Ray calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Smoothed Moving Average (SMMA) function
    def smma(period):
        sma = np.full(len(close_1w), np.nan)
        sma[period-1] = np.mean(close_1w[:period])
        for i in range(period, len(close_1w)):
            sma[i] = (sma[i-1] * (period-1) + close_1w[i]) / period
        return sma
    
    # Williams Alligator components
    jaw = smma(13)  # 13-period SMMA
    teeth = smma(8)  # 8-period SMMA
    lips = smma(5)   # 5-period SMMA
    
    # Apply offsets: Jaw shifted 8 bars, Teeth shifted 5 bars, Lips shifted 3 bars
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # First 8, 5, 3 values will be NaN after roll, handled by alignment
    
    # Elder Ray components
    ema13 = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1w - ema13
    bear_power = ema13 - low_1w
    
    # Elder Ray power signals (rising condition)
    bull_power_rising = np.gradient(bull_power) > 0  # Simple upward slope
    bear_power_rising = np.gradient(bear_power) > 0  # Simple upward slope
    
    # Align Alligator and Elder Ray components to 12h timeframe
    jaw_12h = align_htf_to_ltf(prices, df_1w, jaw_shifted)
    teeth_12h = align_htf_to_ltf(prices, df_1w, teeth_shifted)
    lips_12h = align_htf_to_ltf(prices, df_1w, lips_shifted)
    bull_power_12h = align_htf_to_ltf(prices, df_1w, bull_power)
    bear_power_12h = align_htf_to_ltf(prices, df_1w, bear_power)
    bull_power_rising_12h = align_htf_to_ltf(prices, df_1w, bull_power_rising.astype(float))
    bear_power_rising_12h = align_htf_to_ltf(prices, df_1w, bear_power_rising.astype(float))
    
    # 12h price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(jaw_12h[i]) or np.isnan(teeth_12h[i]) or np.isnan(lips_12h[i]) or \
           np.isnan(bull_power_12h[i]) or np.isnan(bear_power_12h[i]) or \
           np.isnan(bull_power_rising_12h[i]) or np.isnan(bear_power_rising_12h[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Lips > Teeth > Jaw (bullish alignment) AND Bull Power > 0 and rising AND volume surge
            if (lips_12h[i] > teeth_12h[i] > jaw_12h[i] and 
                bull_power_12h[i] > 0 and bull_power_rising_12h[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Lips < Teeth < Jaw (bearish alignment) AND Bear Power > 0 and rising AND volume surge
            elif (lips_12h[i] < teeth_12h[i] < jaw_12h[i] and 
                  bear_power_12h[i] > 0 and bear_power_rising_12h[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Any Alligator condition breaks OR Bear Power becomes positive
            if not (lips_12h[i] > teeth_12h[i] > jaw_12h[i]) or bear_power_12h[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Any Alligator condition breaks OR Bull Power becomes positive
            if not (lips_12h[i] < teeth_12h[i] < jaw_12h[i]) or bull_power_12h[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_ElderRay_Volume"
timeframe = "12h"
leverage = 1.0