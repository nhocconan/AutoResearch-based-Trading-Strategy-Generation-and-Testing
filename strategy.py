# State your hypothesis: 
# This strategy uses the 4-hour Williams Alligator with 1-day Elder Ray Bull/Bear Power
# to identify trend strength and direction. The Alligator (Jaw, Teeth, Lips) acts as
# a dynamic trend filter, while Elder Ray confirms bull/bear power relative to the
# 13-period EMA. A long signal triggers when the Alligator is aligned bullish (Lips > Teeth > Jaw)
# and Bull Power is positive; short when aligned bearish (Lips < Teeth < Jaw) and Bear Power negative.
# Volume confirmation (>1.5x 20-period average) ensures momentum behind the move.
# Exits occur when the Alligator alignment breaks or Elder Ray signal weakens.
# This combines trend-following with momentum confirmation to work in both bull and bear markets
# by adapting to the prevailing trend direction while filtering weak moves.
# Target: 20-50 trades per year to avoid excessive fee churn.

#!/usr/bin/env python3
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
    
    # Load 1-day data ONCE before loop for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Williams Alligator on 4H (13,8,5 SMAs shifted by 8,5,3)
    close_series = pd.Series(close)
    jaw = close_series.rolling(window=13, min_periods=13).mean()
    teeth = close_series.rolling(window=8, min_periods=8).mean()
    lips = close_series.rolling(window=5, min_periods=5).mean()
    jaw = jaw.shift(8)  # Shift jaw by 8 bars
    teeth = teeth.shift(5)  # Shift teeth by 5 bars
    lips = lips.shift(3)  # Shift lips by 3 bars
    jaw = jaw.values
    teeth = teeth.values
    lips = lips.values
    
    # Elder Ray on 1D: Bull Power = High - EMA13, Bear Power = Low - EMA13
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # Align Elder Ray to 4H timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume confirmation: 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (max of Alligator shifts + Elder Ray)
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Bullish Alligator alignment: Lips > Teeth > Jaw
            bullish_alligator = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
            # Bearish Alligator alignment: Lips < Teeth < Jaw
            bearish_alligator = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
            
            # Long setup: Bullish Alligator + positive Bull Power + volume confirmation
            if bullish_alligator and (bull_power_aligned[i] > 0) and (vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: Bearish Alligator + negative Bear Power + volume confirmation
            elif bearish_alligator and (bear_power_aligned[i] < 0) and (vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Alligator alignment breaks bullish OR Bull Power turns negative
            bullish_alligator = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
            if not bullish_alligator or (bull_power_aligned[i] <= 0):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Alligator alignment breaks bearish OR Bear Power turns positive
            bearish_alligator = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
            if not bearish_alligator or (bear_power_aligned[i] >= 0):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Alligator_ElderRay_Volume"
timeframe = "4h"
leverage = 1.0