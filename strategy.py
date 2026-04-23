#!/usr/bin/env python3
"""
Hypothesis: 1d Williams Alligator alignment with 1w trend filter and volume confirmation.
Long when Alligator jaws < teeth < lips (bullish alignment) AND price > lips AND 1w close > 1w open AND volume > 1.5x 20-period average.
Short when Alligator jaws > teeth > lips (bearish alignment) AND price < lips AND 1w close < 1w open AND volume > 1.5x 20-period average.
Exit when price crosses back below/above lips or Alligator alignment breaks.
Uses 1w HTF for trend direction (avoids counter-trend trades). Target: 30-100 total trades over 4 years (7-25/year).
Williams Alligator identifies trending vs ranging markets; alignment ensures we only trade with strong momentum.
Works in both bull (long alignments) and bear (short alignments) markets when trends exist.
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
    
    # Calculate 1d Williams Alligator (13,8,5 SMAs shifted)
    # Jaws: 13-period SMA shifted 8 bars
    # Teeth: 8-period SMA shifted 5 bars  
    # Lips: 5-period SMA shifted 3 bars
    jaws = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Calculate 1w trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1w trend: close > open = bullish, close < open = bearish
    trend_1w = (df_1w['close'].values > df_1w['open'].values).astype(int)  # 1 for bullish, 0 for bearish
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13 + 8, 8 + 5, 5 + 3, 20)  # jaws(21), teeth(13), lips(8), vol(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(trend_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        jaw_val = jaws[i]
        tooth_val = teeth[i]
        lip_val = lips[i]
        trend_val = trend_1w_aligned[i]
        vol_ma_val = vol_ma[i]
        
        # Alligator alignment conditions
        bullish_align = jaw_val < tooth_val < lip_val  # jaws < teeth < lips
        bearish_align = jaw_val > tooth_val > lip_val  # jaws > teeth > lips
        
        if position == 0:
            # Long: Bullish alignment AND price > lips AND 1w bullish trend AND volume spike
            if bullish_align and price > lip_val and trend_val == 1 and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment AND price < lips AND 1w bearish trend AND volume spike
            elif bearish_align and price < lip_val and trend_val == 0 and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price crosses lips OR alignment breaks
            if position == 1:
                exit_condition = (price < lip_val) or not (jaw_val < tooth_val < lip_val)
            else:  # position == -1
                exit_condition = (price > lip_val) or not (jaw_val > tooth_val > lip_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_WilliamsAlligator_Alignment_1wTrend_VolumeConfirmation_LipsExit"
timeframe = "1d"
leverage = 1.0