#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation.
# Uses 1d EMA50 for trend alignment, 12h Williams Alligator (Jaw/Teeth/Lips) for entry signals,
# and volume spike (>2.0x 20-bar avg) for confirmation. Designed for low trade frequency
# (target 50-150 total over 4 years) to minimize fee drag while capturing medium-term trends.
# Works in bull markets via long signals when Lips > Teeth > Jaw and price > 1d EMA50.
# Works in bear markets via short signals when Lips < Teeth < Jaw and price < 1d EMA50.
# Alligator crossover avoids whipsaws, volume confirmation filters false breakouts,
# and 1d trend filter ensures alignment with higher timeframe direction.

name = "12h_WilliamsAlligator_1dEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

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
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator on 12h (primary TF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    median_12h = (df_12h['high'].values + df_12h['low'].values + df_12h['close'].values) / 3.0
    jaw = pd.Series(median_12h).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_12h).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_12h).rolling(window=5, min_periods=5).mean().shift(3).values
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Calculate average volume for confirmation (20-period LTF)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(20, 13), n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Lips > Teeth > Jaw (Alligator bullish alignment), price > 1d EMA50, volume spike (>2.0x avg)
            if (lips_aligned[i] > teeth_aligned[i] and 
                teeth_aligned[i] > jaw_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 2.0 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Lips < Teeth < Jaw (Alligator bearish alignment), price < 1d EMA50, volume spike (>2.0x avg)
            elif (lips_aligned[i] < teeth_aligned[i] and 
                  teeth_aligned[i] < jaw_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 2.0 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close if Alligator turns bearish (Lips < Jaw) or volume drops
            if (lips_aligned[i] < jaw_aligned[i]) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close if Alligator turns bullish (Lips > Jaw) or volume drops
            if (lips_aligned[i] > jaw_aligned[i]) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals