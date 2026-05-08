#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Williams Alligator with Elder Ray for trend confirmation and volume spike for entry timing.
# Williams Alligator (jaw=13, teeth=8, lips=5) defines trend direction via jaw/teeth/lips alignment.
# Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) confirms trend strength.
# Enter long when: Alligator aligned bullish (lips>teeth>jaw) AND Bull Power > 0 AND volume > 1.5x 20-period EMA.
# Enter short when: Alligator aligned bearish (lips<teeth<jaw) AND Bear Power > 0 AND volume > 1.5x 20-period EMA.
# Exit when Alligator alignment breaks or Elder Ray power turns negative.
# Designed for low trade frequency (20-40/year) to avoid fee drag. Works in trending markets via trend-following logic.

name = "4h_12hAlligator_ElderRay_Volume"
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
    
    # Get 12h data for Williams Alligator and Elder Ray
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Williams Alligator (13,8,5 SMMA)
    def smoothed_moving_average(data, period):
        sma = np.zeros_like(data)
        sma[:period] = np.nan
        sma[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            sma[i] = (sma[i-1] * (period-1) + data[i]) / period
        return sma
    
    jaw = smoothed_moving_average(close_12h, 13)  # Blue line (13-period)
    teeth = smoothed_moving_average(close_12h, 8)  # Red line (8-period)
    lips = smoothed_moving_average(close_12h, 5)   # Green line (5-period)
    
    # Calculate Elder Ray Power (EMA13-based)
    ema13 = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_12h - ema13
    bear_power = ema13 - low_12h
    
    # Align 12h indicators to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power)
    
    # Volume confirmation: 4h volume > 1.5x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure enough data for Alligator and Elder Ray
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Alligator bullish alignment + Bull Power positive + volume confirmation
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and 
                bull_power_aligned[i] > 0 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Alligator bearish alignment + Bear Power positive + volume confirmation
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and 
                  bear_power_aligned[i] > 0 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks bearish OR Bull Power turns negative
            if not (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]) or bull_power_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks bullish OR Bear Power turns negative
            if not (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]) or bear_power_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals