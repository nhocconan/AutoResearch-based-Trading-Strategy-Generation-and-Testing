#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Williams Alligator: Jaw (13-period SMMA, 8-bar offset), Teeth (8-period SMMA, 5-bar offset), Lips (5-period SMMA, 3-bar offset)
# Trend: Alligator lines aligned (Jaw > Teeth > Lips for uptrend, reverse for downtrend)
# Entry: Uptrend + Lips cross above Teeth + volume spike = long
# Entry: Downtrend + Lips cross below Teeth + volume spike = short
# Uses discrete sizing 0.25 to target 75-200 total trades over 4 years (19-50/year) for 4h timeframe

name = "4h_WilliamsAlligator_1dEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def smma(source, length):
    """Smoothed Moving Average (SMMA)"""
    if length < 1:
        return np.full_like(source, np.nan, dtype=float)
    result = np.full_like(source, np.nan, dtype=float)
    # First value is simple moving average
    if len(source) >= length:
        result[length-1] = np.mean(source[:length])
    # Subsequent values: SMMA = (PREV_SMMA * (length-1) + CLOSE) / length
    for i in range(length, len(source)):
        result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 trend filter from prior completed 1d bar
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_shifted = np.roll(ema50_1d, 1)
    ema50_1d_shifted[0] = np.nan
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d_shifted)
    
    # Williams Alligator components (using close prices)
    jaw = smma(close, 13)  # Jaw: 13-period SMMA
    teeth = smma(close, 8)  # Teeth: 8-period SMMA
    lips = smma(close, 5)  # Lips: 5-period SMMA
    
    # Apply offsets as per Williams Alligator specification
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Uptrend (Jaw > Teeth > Lips) AND Lips cross above Teeth AND volume spike
            if jaw[i] > teeth[i] and teeth[i] > lips[i] and lips[i] > teeth[i-1] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Downtrend (Lips < Teeth < Jaw) AND Lips cross below Teeth AND volume spike
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and lips[i] < teeth[i-1] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Downtrend begins (Lips < Teeth) OR price closes below Jaw
            if lips[i] < teeth[i] or close[i] < jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Uptrend begins (Lips > Teeth) OR price closes above Jaw
            if lips[i] > teeth[i] or close[i] > jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals